"""Variáveis de entrada e padronização comuns ao HDBSCAN* aprimorado e aos supervisionados."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
# Variáveis criadas por engineer_features, as únicas que build_preprocessor padroniza
DERIVED_COLUMNS = ("log_amount", "hour_sin", "hour_cos")
# Variáveis de engineer_features que nenhum modelo recebe. A hora do dia saiu da distância do
# HDBSCAN* nas rodadas exploratórias e sai também do Random Forest e do XGBoost, para que os três
# modelos partam das mesmas variáveis.
EXCLUDED_FEATURES = ("hour_sin", "hour_cos")


def engineer_features(X):
    """
    Constrói as variáveis de entrada dos modelos a partir das colunas originais da base.

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

        O valor bruto de Time não entra: ao longo dos dois dias da base, ele só ordena as
        transações, como um identificador. O HDBSCAN* aprimorado e os modelos supervisionados
        usam esta mesma função, por meio de model_features, que retira as variáveis de
        EXCLUDED_FEATURES, de modo que todos recebem as mesmas variáveis.

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


def model_features(X, excluded_features=EXCLUDED_FEATURES):
    """
    Deriva as variáveis de engineer_features e retira as que não entram nos modelos.

    Objetivo:
        Monta as variáveis que o HDBSCAN*, o Random Forest e o XGBoost recebem. A exclusão
        acontece na cópia em memória, de modo que os arquivos dos folds não mudam.

    Parâmetros:
        X (pd.DataFrame): transações com as colunas Time, V1 a V28 e Amount, sem a coluna alvo.
        excluded_features (Sequence[str]): variáveis de engineer_features retiradas. O padrão é
            EXCLUDED_FEATURES, a hora do dia.

    Retorno:
        pd.DataFrame com as variáveis de engineer_features, sem as excluídas e com o índice de X.

    Exemplo de uso:
        X_train = model_features(df_train.drop(columns=["Class"]))
        list(X_train.columns)[-1]   # 'log_amount', já sem hour_sin e hour_cos
    """
    return engineer_features(X).drop(columns=list(excluded_features))


def scaled_columns(excluded_features=EXCLUDED_FEATURES, extra_columns=()):
    """
    Lista as variáveis que build_preprocessor padroniza depois de model_features.

    Objetivo:
        Retira das variáveis derivadas, e das extras pedidas, as que model_features excluiu,
        para que o ColumnTransformer não procure colunas ausentes.

    Parâmetros:
        excluded_features (Sequence[str]): as mesmas variáveis passadas a model_features.
        extra_columns (Sequence[str]): outras variáveis padronizadas, como V1 a V28 quando o
            pipeline do HDBSCAN* pede a padronização dos componentes.

    Retorno:
        list[str] na ordem de DERIVED_COLUMNS seguida de extra_columns.

    Exemplo de uso:
        scaled_columns()                           # ['log_amount']
        build_preprocessor(scaled_columns())       # padroniza só log_amount
    """
    return [
        column for column in (*DERIVED_COLUMNS, *extra_columns) if column not in excluded_features
    ]


def build_preprocessor(scaled_columns=DERIVED_COLUMNS):
    """
    Cria, ainda sem ajuste, o transformador que padroniza as variáveis de engineer_features.

    Objetivo:
        O HDBSCAN* e as distâncias aos vizinhos mais próximos usam a distância euclidiana, que é
        sensível à escala de cada variável. O transformador aplica o StandardScaler a log_amount,
        hour_sin e hour_cos, que passam a ter média zero e variância unitária. O logaritmo já
        reduz a assimetria de Amount (de 16,98 para 0,16 na base completa), de modo que o
        RobustScaler, antes usado em log_amount, não alterava o resultado: no fold interno 1, a
        AUPRC da distância kNN mudou só na terceira casa decimal. Os componentes V1 a V28 seguem
        sem alteração (remainder="passthrough"): padronizá-los iguala o peso de componentes de
        variância muito diferente na distância e reduziu essa AUPRC de 0,363 para 0,308. Na
        validação aninhada da base completa, com o GLOSH e sem a hora, a AUPRC média de três
        folds internos caiu de 0,330 para 0,302 com os V padronizados, em todos os
        min_samples de 200 a 400.

        Os modelos supervisionados recebem a mesma padronização, para que os três modelos partam
        das mesmas variáveis. Nas árvores de decisão, que dividem cada variável por limiares, uma
        transformação afim crescente de uma variável não altera as divisões possíveis.

        O ajuste deve usar apenas o conjunto de treinamento (fit_transform), e a validação ou o
        teste recebem somente transform. Assim, médias e desvios vêm exclusivamente do treino,
        sem vazamento de informação. A saída é um np.ndarray sem nomes
        de colunas, na ordem log_amount, hour_sin, hour_cos e V1 a V28, porque o
        ColumnTransformer acrescenta as colunas repassadas ao final.

    Parâmetros:
        scaled_columns (Sequence[str]): variáveis presentes na entrada, padronizadas nesta
            ordem. O padrão são as três derivadas de engineer_features. O pipeline do HDBSCAN*
            passa só as que restam depois de retirar variáveis da distância, como a hora, e
            acrescenta V1 a V28 quando a configuração pede a padronização dos componentes.

    Retorno:
        sklearn.compose.ColumnTransformer ainda não ajustado.

    Exemplo de uso:
        preprocessor = build_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)   # estatísticas calculadas no treino
        X_val_scaled = preprocessor.transform(X_val)           # as mesmas estatísticas na validação
    """
    return ColumnTransformer(
        transformers=[
            ("std_scaler", StandardScaler(), list(scaled_columns)),
        ],
        remainder="passthrough",
    )
