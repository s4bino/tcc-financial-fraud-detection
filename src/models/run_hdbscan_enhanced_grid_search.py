import os
import time
import json
import itertools

import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import approximate_predict
from cuml.neighbors import NearestNeighbors
import dbcv

from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import train_test_split

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def to_numpy(values):
    """
    Converte para np.ndarray, na memória do host, as saídas do cuML e de outras bibliotecas.

    Objetivo:
        O cuML pode devolver rótulos, probabilidades e distâncias como arrays da GPU. A função
        testa três casos, nesta ordem: objetos com o método get(), como os arrays do CuPy, cujo
        get() copia os dados da GPU para a memória principal; objetos que oferecem to_numpy();
        e qualquer outro objeto aceito por np.asarray, como listas e arrays do NumPy. Assim, o
        restante do pipeline opera sempre sobre arrays do NumPy, qualquer que seja o tipo de
        saída configurado no cuML.

        Como pandas.Series e pandas.DataFrame também têm um método get(), que exige uma chave,
        a função não se destina a objetos do pandas; para eles, usar .to_numpy() diretamente.

    Parâmetros:
        values: array da GPU, array do NumPy, sequência ou objeto com to_numpy().

    Retorno:
        np.ndarray com os mesmos valores, na memória do host.

    Exemplo de uso:
        labels = to_numpy(model.labels_)   # rótulos do HDBSCAN* como np.ndarray
        to_numpy([0.2, 0.7])               # array([0.2, 0.7])
    """
    if hasattr(values, "get"):
        return values.get()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def engineer_features(X):
    """
    Constrói as variáveis de entrada do HDBSCAN* aprimorado a partir das colunas originais da base.

    Objetivo:
        Mantém os componentes principais V1 a V28 e substitui Amount e Time por três variáveis
        derivadas. O valor da transação entra como log_amount = ln(1 + Amount), transformação
        que comprime a cauda direita da distribuição, fortemente assimétrica em dados
        financeiros, e permanece definida quando Amount é zero. O tempo é reduzido à posição da
        transação no ciclo de 24 horas, hora = (Time mod 86400) / 3600, e codificado como o par
        (sen(2π·hora/24), cos(2π·hora/24)). A codificação cíclica preserva a proximidade entre
        23h e 0h, que uma escala linear trataria como extremos opostos, e o par seno e cosseno
        identifica cada hora de forma única. Como Time conta os segundos desde a primeira
        transação da base, a hora obtida é relativa a esse instante, e não ao horário do relógio.

    Parâmetros:
        X (pd.DataFrame): transações com as colunas Time, V1 a V28 e Amount, sem a coluna alvo.

    Retorno:
        pd.DataFrame com 31 colunas, na ordem V1 a V28, log_amount, hour_sin e hour_cos, e o
        mesmo índice de X.

    Exemplo de uso:
        X_train = engineer_features(df_train.drop(columns=["Class"]))
        # Time = 90000 s cai na hora 1 do ciclo: hour_sin ≈ 0.259 e hour_cos ≈ 0.966
        # Amount = 99 resulta em log_amount = ln(100) ≈ 4.605
    """
    # fds-et-19-anomaly-detection-models, célula 11, linha 50
    hour = (X["Time"] % 86400) / 3600
    features = X[V_COLUMNS].copy()
    # fds-et-19-anomaly-detection-models, célula 11, linha 48
    features["log_amount"] = np.log1p(X["Amount"])
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return features


def build_preprocessor():
    """
    Cria, ainda sem ajuste, o transformador que padroniza as variáveis derivadas por engineer_features.

    Objetivo:
        O HDBSCAN* e as distâncias aos vizinhos mais próximos usam a distância euclidiana, que é
        sensível à escala de cada variável. O transformador aplica o RobustScaler a log_amount,
        subtraindo a mediana e dividindo pelo intervalo interquartil, estatísticas pouco
        afetadas por valores extremos, e o StandardScaler a hour_sin e hour_cos, que passam a ter
        média zero e variância unitária. Os componentes V1 a V28 seguem sem alteração
        (remainder="passthrough"), por já resultarem da PCA aplicada na origem da base.

        O ajuste deve usar apenas o conjunto de treinamento (fit_transform), e a validação ou o
        teste recebem somente transform. Assim, mediana, quartis, médias e desvios vêm
        exclusivamente do treino, sem vazamento de informação. A saída é um np.ndarray sem nomes
        de colunas, na ordem log_amount, hour_sin, hour_cos e V1 a V28, porque o
        ColumnTransformer acrescenta as colunas repassadas ao final.

    Retorno:
        sklearn.compose.ColumnTransformer ainda não ajustado.

    Exemplo de uso:
        preprocessor = build_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)   # estatísticas calculadas no treino
        X_val_scaled = preprocessor.transform(X_val)           # as mesmas estatísticas na validação
    """
    return ColumnTransformer(
        transformers=[
            ("rob_scaler", RobustScaler(), ["log_amount"]),
            ("std_scaler", StandardScaler(), ["hour_sin", "hour_cos"])
        ],
        remainder="passthrough"
    )


# fds-et-19-anomaly-detection-models, célula 15, linhas 62-65
def neighbor_distances(X_reference, X_query, n_neighbors):
    """
    Calcula, para cada ponto consultado, as distâncias aos n_neighbors vizinhos mais próximos na referência.

    Objetivo:
        Ajusta um NearestNeighbors do cuML, na GPU e com a métrica euclidiana padrão, sobre
        X_reference e consulta os pontos de X_query. A coluna j da saída contém a distância ao
        (j+1)-ésimo vizinho mais próximo, em ordem crescente. No pipeline, a distância ao vizinho
        de ordem min_samples no treino funciona como medida local de densidade, análoga à
        distância núcleo do HDBSCAN* de Campello, Moulavi e Sander (2013): quanto maior a
        distância, mais esparsa a vizinhança e mais anômalo o ponto.

        Quando X_query é o próprio X_reference, cada ponto é o seu vizinho mais próximo, à
        distância zero. Por isso, no treino pede-se um vizinho a mais e lê-se a coluna
        min_samples, enquanto na validação e no teste, cujos pontos não pertencem à referência,
        lê-se a coluna min_samples - 1. As duas leituras correspondem ao mesmo vizinho de ordem
        min_samples, sem contar o próprio ponto.

    Parâmetros:
        X_reference (np.ndarray): pontos que formam o índice de busca, em geral o treino padronizado.
        X_query (np.ndarray): pontos cujas distâncias serão calculadas.
        n_neighbors (int): número de vizinhos retornados por ponto.

    Retorno:
        np.ndarray de forma (len(X_query), n_neighbors), com as distâncias em ordem crescente.

    Exemplo de uso:
        d_train = neighbor_distances(X_train_scaled, X_train_scaled, 301)
        d_train[:, 300]   # distância de cada ponto do treino ao 300º vizinho, sem contar ele mesmo
        d_val = neighbor_distances(X_train_scaled, X_val_scaled, 300)
        d_val[:, 299]     # distância de cada ponto da validação ao seu 300º vizinho no treino
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_reference)
    distances, _ = nn.kneighbors(X_query)
    return to_numpy(distances)


# fds-et-19-anomaly-detection-models, célula 15, linhas 1-4 (lá min-max no teste; aqui posto relativo ao treino)
def percentile_rank(reference, values):
    """
    Converte valores no posto percentil que ocupam na distribuição de referência.

    Objetivo:
        Para cada valor v, retorna a fração dos elementos de reference menores ou iguais a v, isto
        é, a função de distribuição acumulada empírica da referência avaliada em v. O resultado
        fica em [0, 1]: valores abaixo do mínimo da referência recebem 0, e valores iguais ou
        acima do máximo recebem 1. A ordenação custa O(n log n), e cada consulta, O(log n), por
        busca binária (np.searchsorted).

        No pipeline, a referência é sempre a distribuição do treino. Isso leva escores de
        naturezas diferentes, como uma probabilidade e uma distância, a uma escala comum sem usar
        nenhuma estatística da validação ou do teste, ao contrário de uma normalização min-max
        calculada sobre o próprio conjunto avaliado.

    Parâmetros:
        reference (np.ndarray): valores que definem a distribuição, como os escores do treino.
        values (np.ndarray): valores a converter.

    Retorno:
        np.ndarray do mesmo tamanho de values, com postos em [0, 1].

    Exemplo de uso:
        percentile_rank(np.array([1, 2, 3, 4]), np.array([0.5, 2, 3.5, 10]))
        # array([0.  , 0.5 , 0.75, 1.  ])
    """
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


# fds-et-19-anomaly-detection-models, célula 31, linha 10
def ensemble_score(train_scores, scores):
    """
    Combina vários escores de anomalia em um escore único pela média dos postos percentis.

    Objetivo:
        Cada escore de scores é convertido no seu posto percentil em relação ao escore
        correspondente de train_scores (percentile_rank), e o escore combinado de cada ponto é a
        média aritmética desses postos, com pesos iguais. Os dois escores usados no pipeline, 1
        menos a probabilidade de pertinência ao cluster e a distância ao vizinho de ordem
        min_samples, crescem com o grau de anomalia; o resultado, portanto, também cresce com
        ele e fica em [0, 1]. Por operar sobre postos, a combinação independe da escala e da
        forma da distribuição de cada escore, e o valor combinado só se aproxima de 1 quando o
        ponto é extremo, em relação ao treino, nos dois critérios.

    Parâmetros:
        train_scores (list[np.ndarray]): distribuição de cada escore no treino, usada como referência.
        scores (list[np.ndarray]): os mesmos escores, na mesma ordem, para os pontos avaliados
            (o próprio treino, a validação ou o teste).

    Retorno:
        np.ndarray com um escore combinado por ponto, em [0, 1].

    Exemplo de uso:
        train_scores = [np.array([0.1, 0.2, 0.4, 0.9]), np.array([1.0, 2.0, 3.0, 4.0])]
        ensemble_score(train_scores, [np.array([0.95]), np.array([2.5])])
        # array([0.75]): posto 1.0 no primeiro escore e 0.5 no segundo
    """
    return np.mean(
        [percentile_rank(reference, values) for reference, values in zip(train_scores, scores)],
        axis=0
    )


def hdbscan_enhanced_internal_grid_with_tracking(
    internal_folds_dir,
    param_grid,
    target="Class",
    output_dir="results/unsupervised",
    n_processes=None,
    max_dbcv_sample_size=20000,
    alert_rate=0.01,
    resume=False,
    only_folds=None
):
    """
    Executa o ciclo interno da validação cruzada aninhada do HDBSCAN* aprimorado: grid search
    selecionado pelo DBCV, com acompanhamento das métricas de detecção na validação.

    Objetivo:
        Para cada fold interno (fold_k_train.csv e fold_k_val.csv em internal_folds_dir), avalia
        todas as combinações do produto cartesiano de param_grid, que deve conter
        min_cluster_size e min_samples. Em cada fold:

        1. Remove a coluna alvo, elimina as transações duplicadas apenas do treino, deriva as
           variáveis com engineer_features e as padroniza com build_preprocessor ajustado só no
           treino.
        2. Calcula uma única vez as distâncias aos vizinhos mais próximos até o maior min_samples
           da grade, reaproveitadas por todas as combinações.
        3. Para cada combinação pendente, treina o HDBSCAN* do cuML (CAMPELLO; MOULAVI; SANDER,
           2013) com prediction_data=True e calcula sobre o treino o DBCV (MOULAVI et al., 2014),
           índice interno de validação de agrupamentos por densidade, em [-1, 1], que não usa
           rótulos. Por ter custo quadrático, o DBCV é calculado sobre os pontos distintos e,
           acima de max_dbcv_sample_size, sobre uma amostra estratificada pelos rótulos dos
           clusters (random_state=42), ou aleatória simples quando a estratificação não é
           possível. Sem cluster válido ou em caso de erro, o DBCV recebe -1.
        4. Na validação, combina com ensemble_score dois escores de anomalia: 1 menos a
           probabilidade de pertinência dada por approximate_predict e a distância ao vizinho de
           ordem min_samples no treino. O limiar é o quantil 1 - alert_rate do escore combinado
           do treino, e as transações da validação com escore igual ou superior a ele são
           sinalizadas como fraude. Registram-se precisão, recall, F1, AUPRC do escore combinado
           e de cada componente, TP, FP, FN e os tempos de treino e de validação.

        Os rótulos da validação servem apenas ao acompanhamento: a escolha da configuração de cada
        fold usa somente o DBCV, o que preserva o caráter não supervisionado da seleção. Após cada
        combinação, o histórico é regravado em hdbscan_enhanced_grid_results_by_fold.csv, o que
        permite retomar uma execução interrompida. Ao final, a combinação de maior DBCV de cada
        fold é gravada em hdbscan_enhanced_grid_best_summary.csv e best_params_hdbscan_enhanced.json.

    Parâmetros:
        internal_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_val.csv.
        param_grid (dict): listas de valores para min_cluster_size e min_samples.
        target (str): nome da coluna alvo.
        output_dir (str): pasta dos CSVs e do JSON de resultados.
        n_processes (int | None): número de processos usados no cálculo do DBCV.
        max_dbcv_sample_size (int): número máximo de pontos usados no DBCV.
        alert_rate (float): fração esperada de alertas, que define o limiar no escore do treino.
        resume (bool): se True, lê o CSV parcial e pula os pares (fold, combo_id) já avaliados.
            Como combo_id é a posição da combinação no produto cartesiano, a grade precisa ser a
            mesma da execução anterior.
        only_folds (list[int] | None): restringe a execução a estes folds internos.

    Retorno:
        df_folds (pd.DataFrame): uma linha por fold e combinação, com o DBCV e as métricas de validação.
        df_summary (pd.DataFrame): a linha de maior DBCV de cada fold.
        best_params_per_fold (dict): por fold, min_cluster_size, min_samples e best_dbcv_score.
            As chaves são inteiros no retorno e texto no JSON gravado.

    Exemplo de uso:
        param_grid = {"min_cluster_size": [50], "min_samples": [150, 300]}

        df_folds, df_summary, best_params_per_fold = hdbscan_enhanced_internal_grid_with_tracking(
            internal_folds_dir="data/processed/inner_folds",
            param_grid=param_grid,
            output_dir="results/unsupervised",
            n_processes=8,
            max_dbcv_sample_size=28000,
            alert_rate=0.001,
            resume=True
        )

        print(best_params_per_fold[1])   # configuração de maior DBCV no fold interno 1
    """
    os.makedirs(output_dir, exist_ok=True)

    folds = sorted({
        int(f.split("_")[1])
        for f in os.listdir(internal_folds_dir)
        if f.endswith("_train.csv")
    })
    print(f"Folds encontrados para Avaliação (HDBSCAN aprimorado): {folds}")


    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_param_combinations = pd.DataFrame(all_combinations, columns=param_names)

    print(f"Total de combinações HDBSCAN por fold: {len(df_param_combinations)}")

    max_min_samples = int(df_param_combinations["min_samples"].max())

    best_params_per_fold = {}
    all_trials_history = []
    completed_trials = set()

    partial_path = os.path.join(output_dir, "hdbscan_enhanced_grid_results_by_fold.csv")

    if resume and os.path.isfile(partial_path):
        all_trials_history = pd.read_csv(partial_path).to_dict("records")
        completed_trials = {(int(row["fold"]), int(row["combo_id"])) for row in all_trials_history}
        print(f"Retomando de {partial_path}")
        print(f"Combinações já concluídas, que serão puladas: {len(completed_trials)}")

    if only_folds:
        folds = [f for f in folds if f in set(only_folds)]
        print(f"Restrito aos folds: {folds}")

    for fold in folds:
        pending_combos = [
            combo_id for combo_id in df_param_combinations.index
            if (fold, combo_id) not in completed_trials
        ]
        if not pending_combos:
            print(f"\n--- Fold Interno {fold} já concluído, pulando ---")
            continue

        print(f"\n--- Iniciando Grid Search para o Fold Interno {fold} ({len(pending_combos)} combinações pendentes) ---")


        train_path = os.path.join(internal_folds_dir, f"fold_{fold}_train.csv")
        val_path   = os.path.join(internal_folds_dir, f"fold_{fold}_val.csv")

        df_train = pd.read_csv(train_path)
        df_val   = pd.read_csv(val_path)

        # credit-card-fraud-detection-ds, célula 9, linha 2; credit-card-fraud-detection, célula 10, linha 3
        X_train = engineer_features(df_train.drop(columns=[target]).drop_duplicates())
        X_val   = engineer_features(df_val.drop(columns=[target]))
        y_val   = df_val[target]
        del df_train, df_val

        # ==============================================================
        # 2. PRÉ-PROCESSAMENTO
        # ==============================================================
        preprocessor = build_preprocessor()

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)

        t_knn = time.time()
        train_neighbor_distances = neighbor_distances(X_train_scaled, X_train_scaled, max_min_samples + 1)
        val_neighbor_distances = neighbor_distances(X_train_scaled, X_val_scaled, max_min_samples)
        print(f"Distâncias aos {max_min_samples} vizinhos calculadas em {time.time() - t_knn:.1f} s")

        # ==============================================================
        # GRID
        # ==============================================================
        for combo_id, combo in df_param_combinations.iterrows():
            if (fold, combo_id) in completed_trials:
                continue

            params_dict = combo.to_dict()


            min_cluster_size = int(params_dict["min_cluster_size"])
            min_samples = int(params_dict["min_samples"])

            # ==============================================================
            # 3. TREINAMENTO
            # ==============================================================
            t0 = time.time()
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                prediction_data=True
            )
            model.fit(X_train_scaled)
            train_time = time.time() - t0

            labels = to_numpy(model.labels_)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # ==============================================================
            # 4. A MÉTRICA DE OTIMIZAÇÃO: DBCV
            # ==============================================================
            valid_clusters = set(labels) - {-1}
            if len(valid_clusters) >= 1:
                try:
                    X_mat = X_train_scaled.astype(np.float64)

                    _, unique_indices = np.unique(X_mat, axis=0, return_index=True)

                    X_mat_unique = X_mat[unique_indices]
                    labels_unique = labels[unique_indices]

                    if len(X_mat_unique) > max_dbcv_sample_size:
                        try:
                            X_mat_unique, _, labels_unique, _ = train_test_split(
                                X_mat_unique,
                                labels_unique,
                                train_size=max_dbcv_sample_size,
                                random_state=42,
                                stratify=labels_unique
                            )
                        except ValueError:
                            rng = np.random.RandomState(42)
                            sample_idx = rng.choice(len(X_mat_unique), size=max_dbcv_sample_size, replace=False)
                            X_mat_unique = X_mat_unique[sample_idx]
                            labels_unique = labels_unique[sample_idx]

                    if len(set(labels_unique) - {-1}) >= 1:
                        dbcv_score = dbcv.dbcv(X_mat_unique, labels_unique, noise_id=-1, n_processes=n_processes)
                    else:
                        dbcv_score = -1.0

                except Exception as e:
                    print(f"Erro no DBCV: {e}")
                    dbcv_score = -1.0
            else:
                dbcv_score = -1.0

            # ==============================================================
            # 5. AVALIAÇÃO PREDITIVA NA VALIDAÇÃO (Tracking oculto)
            # ==============================================================
            t1 = time.time()
            _, val_probs = approximate_predict(model, X_val_scaled)

            train_scores = [
                1.0 - to_numpy(model.probabilities_),
                train_neighbor_distances[:, min_samples]
            ]
            val_scores = [
                1.0 - to_numpy(val_probs),
                val_neighbor_distances[:, min_samples - 1]
            ]

            train_ensemble = ensemble_score(train_scores, train_scores)
            val_ensemble = ensemble_score(train_scores, val_scores)
            # credit-card-fraud-detection-ds, célula 36, linha 2; credit-card-fraud-detection, célula 55, linhas 2-5
            alert_threshold = np.quantile(train_ensemble, 1.0 - alert_rate)
            val_time = time.time() - t1

            # fds-et-19-anomaly-detection-models, célula 17, linha 3
            preds = (val_ensemble >= alert_threshold).astype(int)

            f1               = f1_score(y_val, preds, zero_division=0)
            prec             = precision_score(y_val, preds, zero_division=0)
            rec              = recall_score(y_val, preds, zero_division=0)
            auprc            = average_precision_score(y_val, val_ensemble)
            auprc_membership = average_precision_score(y_val, val_scores[0])
            auprc_density    = average_precision_score(y_val, val_scores[1])
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

            all_trials_history.append({
                "fold": fold,
                "combo_id": combo_id,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "dbcv_score": dbcv_score,
                "n_clusters": n_clusters,
                "n_noise_points": n_noise,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "auprc": auprc,
                "auprc_membership": auprc_membership,
                "auprc_density": auprc_density,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "train_time": train_time,
                "val_time": val_time
            })

            pd.DataFrame(all_trials_history).to_csv(partial_path, index=False)

            print(
                f"Combo {combo_id} [{combo_id + 1}/{len(df_param_combinations)}] | min_cluster={min_cluster_size} | min_samples={min_samples} | "
                f"clusters={n_clusters} | ruído={n_noise} ({n_noise / len(labels):.1%}) | DBCV={dbcv_score:.4f} | {time.time() - t0:.0f} s\n"
                f"    AUPRC Val={auprc:.4f} (pertinência={auprc_membership:.4f}, kNN={auprc_density:.4f}) | "
                f"Precisão={prec:.3f} | Recall={rec:.3f} | F1 Val={f1:.4f} | TP={tp} FP={fp} FN={fn}"
            )

        del X_train, X_val, y_val, train_neighbor_distances, val_neighbor_distances

        print(f"Fold interno {fold} concluído e gravado.")

    # ==============================================================
    # 6. GERAÇÃO DE RELATÓRIOS E SELEÇÃO DOS MELHORES
    # ==============================================================
    df_folds = pd.DataFrame(all_trials_history)
    df_folds.to_csv(os.path.join(output_dir, "hdbscan_enhanced_grid_results_by_fold.csv"), index=False)

    summary_data = []
    for fold in sorted(int(f) for f in df_folds['fold'].unique()):
        fold_data = df_folds[df_folds['fold'] == fold]

        best_row = fold_data.sort_values("dbcv_score", ascending=False).iloc[0]

        best_params_per_fold[fold] = {
            "min_cluster_size": int(best_row["min_cluster_size"]),
            "min_samples": int(best_row["min_samples"]),
            "best_dbcv_score": float(best_row["dbcv_score"])
        }
        summary_data.append(best_row.to_dict())

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, "hdbscan_enhanced_grid_best_summary.csv"), index=False)

    with open(os.path.join(output_dir, "best_params_hdbscan_enhanced.json"), "w") as f:
        json.dump(best_params_per_fold, f, indent=4)

    return df_folds, df_summary, best_params_per_fold

def hdbscan_enhanced_outer_evaluation(
    outer_folds_dir,
    best_parameter_achieved,
    target="Class",
    output_dir="results/unsupervised",
    alert_rate=0.01
):
    """
    Executa o ciclo externo da validação cruzada aninhada: estima, nos folds externos de teste, o
    desempenho de detecção do HDBSCAN* aprimorado com uma configuração fixa.

    Objetivo:
        Para cada fold externo (fold_k_train.csv e fold_k_test.csv em outer_folds_dir), treina o
        HDBSCAN* com o min_cluster_size e o min_samples de best_parameter_achieved sobre todo o
        treino externo, depois de remover as duplicatas, derivar as variáveis e padronizá-las com
        estatísticas do treino. O teste segue a mesma regra de decisão do ciclo interno: o escore
        combinado de ensemble_score, formado por 1 menos a probabilidade de pertinência de
        approximate_predict e pela distância ao vizinho de ordem min_samples no treino, é
        comparado ao quantil 1 - alert_rate do escore combinado do treino. Os rótulos do teste
        entram apenas no cálculo das métricas.

        Todas as rodadas externas usam a mesma configuração, conforme o critério de seleção
        adotado no trabalho e descrito na Seção 3.1.3.4 da monografia. O relatório traz uma linha
        por fold e uma linha final, "Média", com a média aritmética das métricas entre os folds,
        e é gravado em hdbscan_enhanced_outer_final_report.csv.

    Parâmetros:
        outer_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_test.csv.
        best_parameter_achieved (dict): configuração avaliada, com min_cluster_size e min_samples.
        target (str): nome da coluna alvo.
        output_dir (str): pasta do relatório final.
        alert_rate (float): fração esperada de alertas, que define o limiar no escore do treino.

    Retorno:
        pd.DataFrame com precisão, recall, F1, AUPRC (combinada e por componente), TP, FP, FN e
        tempos de treino e de predição de cada fold, seguido da linha "Média".

    Exemplo de uso:
        best = {"min_cluster_size": 50, "min_samples": 300}

        report = hdbscan_enhanced_outer_evaluation(
            outer_folds_dir="data/processed/outer_folds",
            best_parameter_achieved=best,
            output_dir="results/unsupervised",
            alert_rate=0.001
        )

        print(report[report["fold"] == "Média"][["precision", "recall", "f1", "auprc"]])
    """
    os.makedirs(output_dir, exist_ok=True)
    final_metrics = []

    print("\n--- Iniciando Avaliação Externa (Outer Folds) com Predição ---")

    folds = sorted({
        int(f.split("_")[1])
        for f in os.listdir(outer_folds_dir)
        if f.endswith("_train.csv")
    })

    min_cluster_size = int(best_parameter_achieved["min_cluster_size"])
    min_samples = int(best_parameter_achieved["min_samples"])

    for fold in folds:
        print(f"Avaliando Fold Externo {fold} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")

        train_path = os.path.join(outer_folds_dir, f"fold_{fold}_train.csv")
        test_path = os.path.join(outer_folds_dir, f"fold_{fold}_test.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # credit-card-fraud-detection-ds, célula 9, linha 2; credit-card-fraud-detection, célula 10, linha 3
        X_train = engineer_features(df_train.drop(columns=[target]).drop_duplicates())
        X_test = engineer_features(df_test.drop(columns=[target]))
        y_test = df_test[target]

        preprocessor = build_preprocessor()

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        t0 = time.time()
        model.fit(X_train_scaled)
        train_time = time.time() - t0

        t1 = time.time()
        _, test_probabilities = approximate_predict(model, X_test_scaled)

        train_scores = [
            1.0 - to_numpy(model.probabilities_),
            neighbor_distances(X_train_scaled, X_train_scaled, min_samples + 1)[:, min_samples]
        ]
        test_scores = [
            1.0 - to_numpy(test_probabilities),
            neighbor_distances(X_train_scaled, X_test_scaled, min_samples)[:, min_samples - 1]
        ]

        train_ensemble = ensemble_score(train_scores, train_scores)
        test_ensemble = ensemble_score(train_scores, test_scores)
        # credit-card-fraud-detection-ds, célula 36, linha 2; credit-card-fraud-detection, célula 55, linhas 2-5
        alert_threshold = np.quantile(train_ensemble, 1.0 - alert_rate)
        predict_time = time.time() - t1

        # fds-et-19-anomaly-detection-models, célula 17, linha 3
        preds = (test_ensemble >= alert_threshold).astype(int)

        f1               = f1_score(y_test, preds, zero_division=0)
        prec             = precision_score(y_test, preds, zero_division=0)
        rec              = recall_score(y_test, preds, zero_division=0)
        auprc            = average_precision_score(y_test, test_ensemble)
        auprc_membership = average_precision_score(y_test, test_scores[0])
        auprc_density    = average_precision_score(y_test, test_scores[1])
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        final_metrics.append({
            "fold": fold,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auprc": auprc,
            "auprc_membership": auprc_membership,
            "auprc_density": auprc_density,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "train_time": train_time,
            "predict_time": predict_time
        })

    df_final = pd.DataFrame(final_metrics)
    mean_metrics = df_final.mean().to_dict()
    mean_metrics["fold"] = "Média"
    df_final = pd.concat([df_final, pd.DataFrame([mean_metrics])], ignore_index=True)

    report_path = os.path.join(output_dir, "hdbscan_enhanced_outer_final_report.csv")
    df_final.to_csv(report_path, index=False)
    print(f"\nResultados preditivos finais reportados em: {report_path}")

    return df_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search do HDBSCAN aprimorado com retomada por fold."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="lê o CSV parcial e pula os folds já concluídos"
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="executa apenas estes folds internos, por exemplo: --folds 2 3"
    )
    args = parser.parse_args()

    INTERNAL_FOLDS_DIR = "/content/drive/MyDrive/TCC - DADOS CV/internal_folds"
    OUTER_FOLDS_DIR = "/content/drive/MyDrive/TCC - DADOS CV/folds_output"

    RESULTS_DIR = "/content/drive/MyDrive/RESULTS"

    # fds-et-19-anomaly-detection-models, célula 5, linha 4 (top-1%)
    ALERT_RATE = 0.002

    param_grid_hdbscan = {
        'min_cluster_size': [5, 15, 50],
        'min_samples': [10, 15, 20, 25, 30, 40, 50, 60]
    }


    n_cores_disponiveis = os.cpu_count() or 1
    print(f"Iniciando pipeline... Processos alocados para o DBCV: {n_cores_disponiveis}")

    print("=== ETAPA 1: Grid Search Interno (Treino + Tracking na Validação) ===")
    df_folds, df_summary, best_params_per_fold = hdbscan_enhanced_internal_grid_with_tracking(
        internal_folds_dir=INTERNAL_FOLDS_DIR,
        param_grid=param_grid_hdbscan,
        target="Class",
        output_dir=f"{RESULTS_DIR}/grid_results_hdbscan_enhanced",
        n_processes=n_cores_disponiveis,
        max_dbcv_sample_size=28000,
        alert_rate=ALERT_RATE,
        resume=args.resume,
        only_folds=args.folds
    )

    print("\nMelhores parâmetros encontrados por fold:")
    print(json.dumps(best_params_per_fold, indent=4))

    best_parameter_achieved = max(
        best_params_per_fold.values(),
        key=lambda fold_params: fold_params["best_dbcv_score"]
    )

    print("\nMelhor parâmetro obtido no ciclo interno:")
    print(json.dumps(best_parameter_achieved, indent=4))

    print("\n=== ETAPA 2: Avaliação Externa (Verificando Fraudes com Labels Reais) ===")
    df_final_report = hdbscan_enhanced_outer_evaluation(
        outer_folds_dir=OUTER_FOLDS_DIR,
        best_parameter_achieved=best_parameter_achieved,
        target="Class",
        output_dir=f"{RESULTS_DIR}/final_reports_hdbscan_enhanced",
        alert_rate=ALERT_RATE
    )

    print("\nRelatório Final (Outer Folds):")
    print(df_final_report)
