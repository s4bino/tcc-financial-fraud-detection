"""Escores de anomalia do HDBSCAN* aprimorado (grau de anomalia e GLOSH) e o limiar de alerta."""

import json

import numpy as np

try:
    from cuml.neighbors import NearestNeighbors
except ImportError:  # sem GPU, como nos testes locais: mesma interface, em CPU
    from sklearn.neighbors import NearestNeighbors


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


# fds-et-19-anomaly-detection-models, célula 15, linhas 62-65
def nearest_neighbors(X_reference, X_query, n_neighbors):
    """
    Busca, para cada ponto consultado, os n_neighbors vizinhos mais próximos na referência.

    Objetivo:
        Ajusta um NearestNeighbors, com a métrica euclidiana padrão, sobre X_reference e consulta
        os pontos de X_query. Usa o do cuML, na GPU, quando ele está instalado, e o do
        scikit-learn, em CPU, caso contrário. Os dois têm a mesma interface e devolvem as mesmas
        distâncias. A coluna j da saída corresponde ao (j+1)-ésimo vizinho mais próximo, em
        ordem crescente de distância.

    Parâmetros:
        X_reference (np.ndarray): pontos que formam o índice de busca, em geral o treino
            padronizado.
        X_query (np.ndarray): pontos cujos vizinhos serão buscados.
        n_neighbors (int): número de vizinhos retornados por ponto.

    Retorno:
        distances (np.ndarray): forma (len(X_query), n_neighbors), distâncias em ordem crescente.
        indices (np.ndarray): mesma forma, posição de cada vizinho em X_reference.

    Exemplo de uso:
        distances, indices = nearest_neighbors(X_train_scaled, X_val_scaled, 600)
        indices[:, 0]   # vizinho mais próximo, no treino, de cada ponto da validação
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_reference)
    distances, indices = nn.kneighbors(X_query)
    return to_numpy(distances), to_numpy(indices)


def neighbor_distances(X_reference, X_query, n_neighbors):
    """
    Calcula, para cada ponto consultado, as distâncias aos n_neighbors vizinhos na referência.

    Objetivo:
        Devolve só as distâncias de nearest_neighbors. A coluna j da saída contém a distância ao
        (j+1)-ésimo vizinho mais próximo, em ordem crescente. No grau de anomalia
        (manual_anomaly_score), a distância ao vizinho de ordem min_samples no treino
        funciona como medida local de densidade, análoga à distância núcleo do HDBSCAN* de
        Campello, Moulavi e Sander (2013): quanto maior a distância, mais esparsa a vizinhança e
        mais anômalo o ponto.

        Quando X_query é o próprio X_reference, cada ponto é o seu vizinho mais próximo, à
        distância zero. Por isso, no treino pede-se um vizinho a mais e lê-se a coluna
        min_samples, enquanto na validação e no teste, cujos pontos não pertencem à referência,
        lê-se a coluna min_samples - 1. As duas leituras correspondem ao mesmo vizinho de ordem
        min_samples, sem contar o próprio ponto. A função knn_distances já aplica essa correção.

    Parâmetros:
        X_reference (np.ndarray): pontos que formam o índice de busca, em geral o treino
            padronizado.
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
    return nearest_neighbors(X_reference, X_query, n_neighbors)[0]


def knn_distances(X_train, X_eval, n_neighbors):
    """
    Calcula as distâncias do treino e do conjunto avaliado aos vizinhos de cada ponto no treino.

    Objetivo:
        Chama neighbor_distances duas vezes, com o treino como referência. No treino, pede um
        vizinho a mais e descarta a primeira coluna, a do próprio ponto, à distância zero. Assim,
        nas duas matrizes, a coluna j é a distância ao (j+1)-ésimo vizinho no treino, e a
        distância ao vizinho de ordem min_samples fica na coluna min_samples - 1 tanto no treino
        quanto na validação ou no teste.

    Parâmetros:
        X_train (np.ndarray): treino padronizado, que forma o índice de busca.
        X_eval (np.ndarray): validação ou teste, padronizado com as estatísticas do treino.
        n_neighbors (int): número de vizinhos por ponto, em geral o maior min_samples da grade.

    Retorno:
        train_distances (np.ndarray): forma (len(X_train), n_neighbors).
        distances (np.ndarray): forma (len(X_eval), n_neighbors).

    Exemplo de uso:
        train_distances, val_distances = knn_distances(X_train_scaled, X_val_scaled, 500)
        val_distances[:, 299]   # distância de cada ponto da validação ao seu 300º vizinho no treino
    """
    train_distances = neighbor_distances(X_train, X_train, n_neighbors + 1)[:, 1:]
    return train_distances, neighbor_distances(X_train, X_eval, n_neighbors)


# fds-et-19-anomaly-detection-models, célula 15, linhas 1-4
# (lá min-max no teste; aqui posto relativo ao treino)
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
        train_scores (list[np.ndarray]): distribuição de cada escore no treino, usada como
            referência.
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
        [
            percentile_rank(reference, values)
            for reference, values in zip(train_scores, scores, strict=True)
        ],
        axis=0,
    )


def manual_anomaly_score(
    X_train,
    X_eval,
    train_probabilities,
    probabilities,
    min_samples,
    neighbors=None,
    component="membership",
    tree=None,
    tree_neighbors=None,
):
    """
    Calcula o grau de anomalia do HDBSCAN* aprimorado no treino e no conjunto avaliado.

    Objetivo:
        Reúne as etapas do grau de anomalia a(x) da Seção 4.4.2 da monografia. São dois
        componentes, que crescem com o grau de anomalia: um escore do agrupamento, escolhido por
        component, e d(x), a distância ao vizinho de ordem min_samples no treino, sem contar o
        próprio ponto (knn_distances). Cada componente é convertido no posto percentil em
        relação ao treino (percentile_rank), e o grau é a média dos dois postos
        (ensemble_score). O treino recebe o seu próprio grau, que define o limiar de alerta, e o
        conjunto avaliado, validação ou teste, é posto na mesma escala.

        O escore do agrupamento tem duas opções. Com component="membership", a da monografia, é
        1 - p(x), em que p(x) é a probabilidade de pertinência ao agrupamento. Como p(x) é zero
        em todo ponto de ruído, esse componente empata todo o ruído no valor máximo. Com
        component="glosh", é o GLOSH (glosh_anomaly_score), que gradua também os pontos de
        ruído, pela densidade em que cada um deixa o seu último agrupamento na árvore condensada.
        Nesse caso, train_probabilities e probabilities não são usados e podem ser None.

        As distâncias aos vizinhos são a etapa mais cara. Quando várias configurações usam o
        mesmo treino, como na grade de min_samples, neighbors recebe o resultado de knn_distances
        e tree_neighbors o de glosh_neighbors, calculados uma vez para o maior min_samples, e a
        função só lê as colunas de cada configuração. Sem eles, as buscas são feitas aqui.

    Parâmetros:
        X_train (np.ndarray): treino padronizado, o mesmo do ajuste do HDBSCAN*.
        X_eval (np.ndarray): validação ou teste, padronizado com as estatísticas do treino.
        train_probabilities: probabilidade de pertinência de cada ponto do treino
            (probabilities_ do modelo ajustado). Só com component="membership".
        probabilities: a mesma probabilidade para o conjunto avaliado (approximate_predict). Só
            com component="membership".
        min_samples (int): o min_samples do HDBSCAN*, que define o vizinho do componente d(x).
        neighbors (tuple | None): par (treino, avaliado) devolvido por knn_distances, com ao
            menos min_samples vizinhos.
        component (str): escore do agrupamento, "membership" (1 - p(x)) ou "glosh".
        tree: árvore condensada do HDBSCAN* ajustado em X_train, como
            model.condensed_tree_.to_numpy() no cuML. Só com component="glosh".
        tree_neighbors (tuple | None): trio devolvido por glosh_neighbors, com min_samples igual
            ou maior. Só com component="glosh".

    Retorno:
        train_degree (np.ndarray): grau de anomalia de cada ponto do treino, em [0, 1].
        degree (np.ndarray): grau de anomalia de cada ponto avaliado, em [0, 1].
        components (list[np.ndarray]): os dois componentes do conjunto avaliado, o escore do
            agrupamento e d(x), para acompanhar a AUPRC de cada um.

    Exemplo de uso:
        _, val_probs = approximate_predict(model, X_val_scaled)
        train_degree, val_degree, val_components = manual_anomaly_score(
            X_train_scaled, X_val_scaled, model.probabilities_, val_probs, min_samples=300
        )
        alert_threshold = np.quantile(train_degree, 1 - 0.001)   # limiar no treino
        alerts = val_degree >= alert_threshold

        train_degree, val_degree, val_components = manual_anomaly_score(   # com o GLOSH
            X_train_scaled, X_val_scaled, None, None, min_samples=300,
            component="glosh", tree=model.condensed_tree_.to_numpy(),
        )
    """
    if component == "membership":
        if train_probabilities is None or probabilities is None:
            raise ValueError("component='membership' exige train_probabilities e probabilities")
        train_cluster_score = 1.0 - to_numpy(train_probabilities)
        cluster_score = 1.0 - to_numpy(probabilities)
    elif component == "glosh":
        if tree is None:
            raise ValueError("component='glosh' exige a árvore condensada em tree")
        train_cluster_score, cluster_score = glosh_anomaly_score(
            tree, X_train, X_eval, min_samples, neighbors=tree_neighbors
        )
    else:
        raise ValueError(f"component deve ser 'membership' ou 'glosh', não {component!r}")

    if neighbors is None:
        neighbors = knn_distances(X_train, X_eval, min_samples)
    train_distances, distances = neighbors

    train_components = [train_cluster_score, train_distances[:, min_samples - 1]]
    components = [cluster_score, distances[:, min_samples - 1]]
    return (
        ensemble_score(train_components, train_components),
        ensemble_score(train_components, components),
        components,
    )


def _index_condensed_tree(tree):
    """
    Reorganiza a árvore condensada em arrays indexados por ponto e por agrupamento.

    Retorna quatro arrays, nesta ordem. Os dois primeiros dão, para cada ponto, o agrupamento de
    que ele sai e a densidade λ da saída. Os dois últimos dão o pai e a densidade de nascimento
    de cada agrupamento, indexados por rótulo menos o rótulo da raiz. A raiz tem pai -1 e nasce
    em λ = 0.
    """
    parent = np.asarray(tree["parent"], dtype=np.intp)
    child = np.asarray(tree["child"], dtype=np.intp)
    lambda_val = np.asarray(tree["lambda_val"], dtype=np.float64)

    root = parent.min()
    is_point = child < root
    if np.count_nonzero(is_point) != root or len(np.unique(child[is_point])) != root:
        raise ValueError("a árvore condensada deve ter uma linha por ponto, rotulados de 0 a n - 1")

    point_cluster = np.empty(root, dtype=np.intp)
    point_lambda = np.empty(root, dtype=np.float64)
    point_cluster[child[is_point]] = parent[is_point] - root
    point_lambda[child[is_point]] = lambda_val[is_point]

    n_clusters = max(parent.max(), child.max()) - root + 1
    cluster_parent = np.full(n_clusters, -1, dtype=np.intp)
    cluster_birth = np.zeros(n_clusters, dtype=np.float64)
    cluster_parent[child[~is_point] - root] = parent[~is_point] - root
    cluster_birth[child[~is_point] - root] = lambda_val[~is_point]
    return point_cluster, point_lambda, cluster_parent, cluster_birth


def _max_lambdas(point_cluster, point_lambda, cluster_parent):
    """λ_max de cada agrupamento: a maior densidade de saída entre os pontos da sua subárvore."""
    max_lambdas = np.zeros(len(cluster_parent))
    np.maximum.at(max_lambdas, point_cluster, point_lambda)
    # Cada passada leva o máximo um nível acima, e o número de passadas é a altura da árvore.
    children = np.flatnonzero(cluster_parent >= 0)
    while True:
        propagated = max_lambdas.copy()
        np.maximum.at(propagated, cluster_parent[children], max_lambdas[children])
        if np.array_equal(propagated, max_lambdas):
            return max_lambdas
        max_lambdas = propagated


def _glosh(lambdas, max_lambdas):
    """Equação (8) de Campello et al. (2015), 1 - λ/λ_max, com os casos de λ infinito ou nulo."""
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 1.0 - lambdas / max_lambdas
    # λ infinito (ponto repetido) é o mais interno possível, e λ_max nulo não tem referência
    scores[np.isinf(lambdas) | (max_lambdas == 0)] = 0.0
    return scores


def cluster_max_lambdas(tree):
    """
    Calcula a maior densidade alcançada por cada agrupamento ou por algum subagrupamento dele.

    Objetivo:
        Na hierarquia do HDBSCAN*, a densidade de um nível é λ = 1/ε, o inverso do raio ε. Um
        agrupamento perde pontos à medida que λ cresce, até que todos os seus pontos e os dos
        seus subagrupamentos virem ruído. A densidade em que isso ocorre, λ_max, é a de que o
        GLOSH precisa como referência (CAMPELLO et al., 2015, Algoritmo 4, p. 5:26): o menor raio
        em que o agrupamento ou algum subagrupamento ainda existe é ε_max = 1/λ_max.

        A função toma, em cada agrupamento, a maior densidade de saída entre os seus próprios
        pontos e a propaga para cima, de cada agrupamento para o pai, até a raiz. A propagação
        não depende da ordem das linhas nem da numeração dos agrupamentos.

    Parâmetros:
        tree: árvore condensada do HDBSCAN* no formato da biblioteca hdbscan, com os campos
            parent, child e lambda_val, como o array estruturado de condensed_tree_.to_numpy() ou
            o DataFrame de condensed_tree_.to_pandas(). Os pontos são rotulados de 0 a n - 1, a
            raiz é n, e os demais agrupamentos recebem rótulos maiores. Cada linha registra um
            ponto que sai do agrupamento parent na densidade lambda_val, ou um agrupamento child
            que nasce de parent nessa densidade.

    Retorno:
        np.ndarray com o λ_max de cada agrupamento, na posição rótulo - n.

    Exemplo de uso:
        tree = {
            "parent": [5, 5, 6, 6, 7, 7, 5],
            "child": [6, 7, 0, 1, 2, 3, 4],
            "lambda_val": [0.5, 0.5, 2.0, 4.0, 1.0, 1.0, 0.25],
        }
        cluster_max_lambdas(tree)
        # array([4., 4., 1.]): a raiz (5) herda o λ_max = 4 do agrupamento 6
    """
    point_cluster, point_lambda, cluster_parent, _ = _index_condensed_tree(tree)
    return _max_lambdas(point_cluster, point_lambda, cluster_parent)


def glosh_scores(tree):
    """
    Calcula o GLOSH de cada ponto usado no ajuste do HDBSCAN*, a partir da árvore condensada.

    Objetivo:
        O GLOSH (Global-Local Outlier Score from Hierarchies) de Campello et al. (2015, Equação
        8, p. 5:25) compara a densidade em que um ponto x deixa o último agrupamento a que
        pertence na hierarquia, λ(x), com a maior densidade alcançada por esse agrupamento ou por
        algum subagrupamento, λ_max(x):

            GLOSH(x) = (λ_max(x) - λ(x)) / λ_max(x) = 1 - ε_max(x) / ε(x)

        O escore fica em [0, 1]. Vale perto de 0 para os pontos da região mais densa do
        agrupamento e se aproxima de 1 para os pontos ligados a ele só em densidades muito
        menores. A referência de cada ponto é o seu próprio agrupamento, e não a base inteira nem
        uma vizinhança de raio fixo, de modo que o escore detecta outliers globais e locais
        (CAMPELLO et al., 2015, p. 5:24). Ao contrário da probabilidade de pertinência, que é zero
        em todo o ruído, o GLOSH ordena também os pontos de ruído: cada um é comparado com o
        agrupamento do qual saiu, e quem sai da raiz é comparado com a base inteira.

        Na árvore condensada, cada ponto aparece em uma única linha, a da sua saída: parent é o
        último agrupamento e lambda_val é λ(x). Com o λ_max de cluster_max_lambdas, o cálculo
        segue o Algoritmo 4 do artigo, em O(n). Um ponto em densidade infinita, que só ocorre com
        pontos repetidos, recebe 0.

    Parâmetros:
        tree: árvore condensada do HDBSCAN*, no formato descrito em cluster_max_lambdas.

    Retorno:
        np.ndarray com o GLOSH de cada ponto, na ordem dos rótulos 0 a n - 1, em [0, 1].

    Exemplo de uso:
        tree = {
            "parent": [5, 5, 6, 6, 7, 7, 5],
            "child": [6, 7, 0, 1, 2, 3, 4],
            "lambda_val": [0.5, 0.5, 2.0, 4.0, 1.0, 1.0, 0.25],
        }
        glosh_scores(tree)
        # array([0.5   , 0.    , 0.    , 0.    , 0.9375])
        # o ponto 0 sai do agrupamento 6 em λ = 2, metade do λ_max = 4 do agrupamento
        # o ponto 4 sai da raiz em λ = 0.25 e é comparado com o λ_max = 4 da base inteira
    """
    point_cluster, point_lambda, cluster_parent, _ = _index_condensed_tree(tree)
    max_lambdas = _max_lambdas(point_cluster, point_lambda, cluster_parent)
    return _glosh(point_lambda, max_lambdas[point_cluster])


def glosh_new_points(
    tree, train_core_distances, core_distances, neighbor_indices, neighbor_distances
):
    """
    Calcula o GLOSH de pontos que não participaram do ajuste, como os da validação e do teste.

    Objetivo:
        Insere cada ponto novo x na hierarquia do treino, sem alterá-la, e aplica a Equação (8)
        de Campello et al. (2015) ao nível em que ele se liga a ela. Pela Definição 3.2 do
        artigo (p. 5:10), x e um ponto y do treino ficam ligados a partir do raio dado pela
        distância de alcançabilidade mútua

            d_mreach(x, y) = max{d_core(x), d_core(y), d(x, y)},

        em que d_core é a distância núcleo (Definição 3.1, p. 5:9). O ponto y, por sua vez,
        pertence ao seu último agrupamento até a densidade λ(y) registrada na árvore. Pelo
        caminho que passa por y, portanto, x fica nesse agrupamento até a densidade

            λ_y(x) = min{1 / d_mreach(x, y), λ(y)}.

        O ponto x deixa o último agrupamento a que pertence na maior dessas densidades entre os
        vizinhos consultados, λ(x) = max_y λ_y(x). Esse agrupamento é o do vizinho y* que
        realiza o máximo ou, se ele nasce numa densidade maior que λ(x), o ancestral mais
        próximo que já existe em λ(x). O escore é então (λ_max - λ(x)) / λ_max, com o λ_max
        desse agrupamento, como no Algoritmo 4 (p. 5:26).

        Como λ(x) ≤ λ(y*) ≤ λ_max, o escore fica em [0, 1], e um ponto idêntico a um ponto do
        treino recebe o mesmo GLOSH que ele. A busca percorre só os vizinhos informados. Como
        d_mreach(x, y) ≥ d(x, y), nenhum ponto além do último vizinho consultado, à distância
        d_k, alcança densidade maior que 1/d_k, e o resultado é exato sempre que λ(x) ≥ 1/d_k.
        Por isso convém consultar mais vizinhos que min_samples; a biblioteca hdbscan, na função
        equivalente (approximate_predict_scores), consulta 2·min_samples.

        As distâncias núcleo do treino e dos pontos novos devem seguir a mesma convenção do
        modelo ajustado, contando ou não o próprio ponto entre os vizinhos, e as dos pontos
        novos são calculadas em relação ao treino.

    Parâmetros:
        tree: árvore condensada do HDBSCAN* ajustado no treino, no formato descrito em
            cluster_max_lambdas.
        train_core_distances (np.ndarray): distância núcleo de cada ponto do treino, na ordem dos
            rótulos da árvore.
        core_distances (np.ndarray): distância núcleo de cada ponto novo, em relação ao treino.
        neighbor_indices (np.ndarray): forma (n_novos, k), índices no treino dos k vizinhos mais
            próximos de cada ponto novo.
        neighbor_distances (np.ndarray): forma (n_novos, k), distâncias a esses vizinhos.

    Retorno:
        np.ndarray com o GLOSH de cada ponto novo, em [0, 1].

    Exemplo de uso:
        tree = {
            "parent": [5, 5, 6, 6, 7, 7, 5],
            "child": [6, 7, 0, 1, 2, 3, 4],
            "lambda_val": [0.5, 0.5, 2.0, 4.0, 1.0, 1.0, 0.25],
        }
        train_core_distances = np.array([0.5, 0.25, 1.0, 1.0, 4.0])
        glosh_new_points(
            tree,
            train_core_distances,
            core_distances=np.array([0.5, 9.0]),
            neighbor_indices=np.array([[0, 1], [4, 2]]),
            neighbor_distances=np.array([[0.0, 0.3], [8.0, 9.0]]),
        )
        # array([0.5   , 0.9722])
        # o primeiro ponto coincide com o ponto 0 do treino e recebe o mesmo GLOSH, 0.5
        # o segundo se liga em λ = 1/9, antes de nascerem os agrupamentos 6 e 7, e é comparado
        # com o λ_max = 4 da raiz: 1 - (1/9)/4 ≈ 0.9722
    """
    point_cluster, point_lambda, cluster_parent, cluster_birth = _index_condensed_tree(tree)
    max_lambdas = _max_lambdas(point_cluster, point_lambda, cluster_parent)

    neighbor_indices = np.asarray(neighbor_indices, dtype=np.intp)
    mreach = np.maximum(
        np.maximum(neighbor_distances, np.asarray(train_core_distances)[neighbor_indices]),
        np.asarray(core_distances)[:, np.newaxis],
    )
    with np.errstate(divide="ignore"):
        link_lambdas = np.minimum(1.0 / mreach, point_lambda[neighbor_indices])

    best = np.argmax(link_lambdas, axis=1)
    rows = np.arange(len(best))
    lambdas = link_lambdas[rows, best]
    clusters = point_cluster[neighbor_indices[rows, best]]
    # Sobe até o ancestral que já existe em λ(x). A raiz nasce em λ = 0 e encerra a subida.
    climb = cluster_birth[clusters] > lambdas
    while climb.any():
        clusters[climb] = cluster_parent[clusters[climb]]
        climb = cluster_birth[clusters] > lambdas
    return _glosh(lambdas, max_lambdas[clusters])


def glosh_neighbors(X_train, X_eval, min_samples):
    """
    Busca os vizinhos de que o GLOSH precisa, no treino e no conjunto avaliado.

    Objetivo:
        O GLOSH dos pontos novos (glosh_new_points) usa as distâncias núcleo do treino e dos
        pontos novos e os vizinhos de cada ponto novo no treino. A função faz as duas buscas de
        uma vez. No treino, consulta min_samples vizinhos do próprio treino, com o próprio
        ponto, à distância zero, na primeira coluna. É a mesma busca com que o HDBSCAN do cuML
        calcula a distância núcleo (_compute_core_dists, em
        cpp/src/hdbscan/detail/reachability.cuh, versão 26.02), lendo a coluna min_samples - 1,
        de acordo com a Definição 3.1 de Campello et al. (2015, p. 5:9). No conjunto avaliado,
        consulta 2·min_samples vizinhos no treino, como a biblioteca hdbscan faz em
        approximate_predict_scores.

        Na grade, a busca é feita uma vez, para o maior min_samples, e glosh_anomaly_score lê as
        colunas de cada configuração.

    Parâmetros:
        X_train (np.ndarray): treino padronizado, o mesmo do ajuste do HDBSCAN*.
        X_eval (np.ndarray): validação ou teste, padronizado com as estatísticas do treino.
        min_samples (int): o min_samples do HDBSCAN*, ou o maior da grade.

    Retorno:
        train_distances (np.ndarray): forma (len(X_train), min_samples), com o próprio ponto na
            primeira coluna.
        distances (np.ndarray): forma (len(X_eval), 2·min_samples).
        indices (np.ndarray): mesma forma, posição de cada vizinho no treino.

    Exemplo de uso:
        neighbors = glosh_neighbors(X_train_scaled, X_val_scaled, 500)   # maior min_samples
    """
    train_distances = neighbor_distances(X_train, X_train, min_samples)
    return (train_distances, *nearest_neighbors(X_train, X_eval, 2 * min_samples))


def glosh_anomaly_score(tree, X_train, X_eval, min_samples, neighbors=None):
    """
    Calcula o GLOSH do treino e do conjunto avaliado a partir do HDBSCAN* ajustado no treino.

    Objetivo:
        Reúne as etapas do GLOSH para o pipeline. O treino recebe o escore da própria árvore
        condensada (glosh_scores), e o conjunto avaliado, validação ou teste, o da inserção na
        hierarquia do treino (glosh_new_points). As distâncias núcleo seguem a convenção do
        HDBSCAN*, que conta o próprio ponto entre os min_samples vizinhos (Definição 3.1 de
        Campello et al., 2015). No treino, é a distância da coluna min_samples - 1 de
        glosh_neighbors, que tem o próprio ponto na coluna 0. Um ponto novo não pertence ao
        treino, e, contando ele mesmo como primeiro vizinho, a sua distância núcleo é a do
        (min_samples - 1)-ésimo vizinho no treino.

        Quando várias configurações usam o mesmo treino, neighbors recebe o resultado de
        glosh_neighbors para o maior min_samples, e a função só lê as colunas de cada
        configuração, sem usar X_train e X_eval. Sem neighbors, a busca é feita aqui.

    Parâmetros:
        tree: árvore condensada do HDBSCAN* ajustado em X_train, no formato descrito em
            cluster_max_lambdas, como model.condensed_tree_.to_numpy() no cuML.
        X_train (np.ndarray): treino padronizado, o mesmo do ajuste.
        X_eval (np.ndarray): validação ou teste, padronizado com as estatísticas do treino.
        min_samples (int): o min_samples do HDBSCAN*, pelo menos 2.
        neighbors (tuple | None): trio devolvido por glosh_neighbors com min_samples igual ou
            maior.

    Retorno:
        train_scores (np.ndarray): GLOSH de cada ponto do treino, em [0, 1].
        scores (np.ndarray): GLOSH de cada ponto avaliado, em [0, 1].

    Exemplo de uso:
        model = HDBSCAN(min_cluster_size=50, min_samples=300).fit(X_train_scaled)
        train_scores, val_scores = glosh_anomaly_score(
            model.condensed_tree_.to_numpy(), X_train_scaled, X_val_scaled, min_samples=300
        )
        alerts = val_scores >= np.quantile(train_scores, 1 - 0.001)   # limiar no treino
    """
    if min_samples < 2:
        raise ValueError("min_samples deve ser pelo menos 2")
    if neighbors is None:
        neighbors = glosh_neighbors(X_train, X_eval, min_samples)
    train_distances, distances, indices = neighbors

    n_query = 2 * min_samples
    scores = glosh_new_points(
        tree,
        train_core_distances=train_distances[:, min_samples - 1],
        core_distances=distances[:, min_samples - 2],
        neighbor_indices=indices[:, :n_query],
        neighbor_distances=distances[:, :n_query],
    )
    return glosh_scores(tree), scores


def read_legitimate_weight(info_path):
    """
    Lê do JSON do fold o peso de cada transação legítima do treino no limiar de alerta.

    Objetivo:
        A subamostragem de src/preprocessing/undersampling.py mantém todas as fraudes do treino
        e sorteia uma fração das legítimas, sem reposição. O sorteio depende só da classe e,
        por isso, não muda a distribuição das variáveis dentro de cada classe (DAL POZZOLO et
        al., 2015, p. 2): as legítimas mantidas representam as descartadas. O peso é o número
        de legítimas do treino original que cada uma representa, a razão entre
        num_legitimate_before e num_legitimate_after do bloco "undersampling" do JSON, ou seja,
        o inverso da proporção de legítimas mantidas neste treino. Essa proporção é a estimativa
        de β, a probabilidade de uma legítima ser selecionada (DAL POZZOLO et al., 2015, p. 3).
        Na base completa, o JSON não tem esse bloco, e o peso é 1.

    Parâmetros:
        info_path (str | Path): JSON gravado na geração dos folds, fold_k_internal_info.json no
            ciclo interno e fold_k_info.json no externo.

    Retorno:
        float: 1.0 na base completa e cerca de 2,84 na subamostrada com 100 mil legítimas.

    Exemplo de uso:
        weight = read_legitimate_weight("data/undersampled/inner_folds/fold_1_internal_info.json")
        # 181962 legítimas antes e 64086 depois: weight ≈ 2.839
    """
    with open(info_path, encoding="utf-8") as file:
        undersampling = json.load(file).get("undersampling")
    if undersampling is None:
        return 1.0
    return undersampling["num_legitimate_before"] / undersampling["num_legitimate_after"]


def alert_threshold(train_scores, alert_rate, train_labels=None, legitimate_weight=1.0):
    """
    Calcula o limiar de alerta, o escore do treino acima do qual fica a fração alert_rate.

    Objetivo:
        Com legitimate_weight igual a 1, o caso da base completa, o limiar é o quantil
        1 - alert_rate do escore do treino, com a interpolação linear do NumPy.

        Na base subamostrada, o treino tem, para cada legítima, cerca de 2,84 vezes as fraudes
        da validação e do teste. Como as fraudes ficam entre os maiores escores, elas ocupariam
        boa parte da fração alert_rate, o limiar subiria, e a validação e o teste, que mantêm a
        proporção real de fraudes, receberiam menos alertas do que alert_rate. Sem correção, a
        subamostragem desloca a fronteira de decisão (DAL POZZOLO et al., 2015, p. 4), e o
        limiar corrigido é o do treino antes da subamostragem (DAL POZZOLO et al., 2015, p. 5).
        Aqui, a correção é feita com pesos: cada legítima pesa legitimate_weight, cada fraude
        pesa 1, porque todas foram mantidas, e o limiar é o quantil ponderado do NumPy pelo
        método inverted_cdf, o menor escore em que a soma dos pesos até ele alcança
        1 - alert_rate do peso total. Com os pesos, o treino volta, aproximadamente, ao tamanho
        e à proporção de fraudes de antes da subamostragem, e alert_rate volta a ser a fração de
        alertas na proporção real. Os rótulos do treino entram só nos pesos, como já entraram
        na subamostragem, e os da validação e do teste ficam de fora. O quantil com pesos exige
        o NumPy 2.0 ou posterior.

    Parâmetros:
        train_scores (np.ndarray): escore de anomalia de cada ponto do treino.
        alert_rate (float): fração esperada de alertas, entre 0 e 1.
        train_labels (array-like | None): rótulo de cada ponto do treino, 1 para fraude, na
            ordem de train_scores. Obrigatório quando legitimate_weight é diferente de 1.
        legitimate_weight (float): peso de cada legítima, de read_legitimate_weight.

    Retorno:
        float: o limiar. As transações avaliadas com escore igual ou superior a ele são
        sinalizadas como fraude.

    Exemplo de uso:
        weight = read_legitimate_weight("data/undersampled/inner_folds/fold_1_internal_info.json")
        threshold = alert_threshold(train_glosh, 0.001, y_train, weight)
        alerts = val_glosh >= threshold
    """
    if legitimate_weight == 1.0:
        return np.quantile(train_scores, 1.0 - alert_rate)
    if train_labels is None:
        raise ValueError("o limiar ponderado exige train_labels")
    weights = np.where(np.asarray(train_labels) == 1, 1.0, legitimate_weight)
    return np.quantile(train_scores, 1.0 - alert_rate, weights=weights, method="inverted_cdf")
