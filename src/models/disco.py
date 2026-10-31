"""DISCO, índice interno de validação de agrupamentos por densidade que também avalia o ruído.

Implementa as Equações 1 a 8 de Beer et al. (2025) sem montar a matriz n x n das distâncias de
conectividade por densidade (dc): as somas e os mínimos dessas distâncias vêm direto da árvore de
fusões. Os resultados coincidem com os do código oficial dos autores, e o tempo e a memória estão
medidos em docs/metrics/disco.md.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

NOISE_LABEL = -1
# Beer et al. (2025, Seção 4.5) fixam μ = 5 em todos os experimentos, por o índice ser pouco
# sensível a esse valor.
DEFAULT_MIN_POINTS = 5


class DCTree:
    """
    Árvore de fusões da distância dc, que depende só dos pontos e de min_points, não dos rótulos.

    Objetivo:
        A distância dc entre dois pontos é a maior aresta do caminho entre eles na árvore geradora
        mínima da distância de alcançabilidade mútua (Equação 1 de Beer et al., 2025). Ao fundir
        as arestas em ordem crescente, a distância dc entre dois pontos é a altura da fusão que
        os une pela primeira vez. A árvore guarda essas fusões e responde em O(n), para cada
        ponto, à soma e ao mínimo das distâncias a um conjunto de pontos, sem a matriz n x n.

        Como não depende dos rótulos, a mesma árvore serve a todas as partições de um mesmo X. Na
        busca em grade, basta construí-la uma vez por fold.

    Parâmetros:
        X (np.ndarray): pontos, com forma (n_pontos, n_variáveis).
        min_points (int): parâmetro μ da distância de núcleo. O próprio ponto conta como o seu
            primeiro vizinho, como no código oficial.

    Atributos:
        n_points (int): número de pontos.
        min_points (int): valor de μ usado na construção.
        core_distances (np.ndarray): distância de núcleo de cada ponto.

    Exemplo de uso:
        tree = DCTree(X_train)                  # constrói uma vez por fold
        disco_score(X_train, labels, tree=tree) # reaproveita em cada partição
    """

    def __init__(self, X, min_points=DEFAULT_MIN_POINTS):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X precisa ter forma (n_pontos, n_variáveis), com ao menos um ponto.")
        if not 1 <= min_points <= len(X):
            raise ValueError(f"min_points={min_points} precisa estar entre 1 e {len(X)}.")

        self.n_points = len(X)
        self.min_points = min_points
        self.core_distances = self._compute_core_distances(X, min_points)
        self._build_merge_tree(*self._prim_edges(X, self.core_distances))

    @staticmethod
    def _compute_core_distances(X, min_points):
        """Devolve a distância de cada ponto ao seu min_points-ésimo vizinho, contando ele mesmo."""
        # A busca "brute", embora mais rápida, usa um atalho numérico em que a distância de um ponto
        # a si mesmo chega a 1e-6, o que distorceria os pontos duplicados. A KD-tree é exata.
        search = NearestNeighbors(n_neighbors=min_points, algorithm="kd_tree", n_jobs=-1).fit(X)
        return search.kneighbors(X)[0][:, -1]

    @staticmethod
    def _prim_edges(X, core_distances):
        """
        Calcula a árvore geradora mínima da alcançabilidade mútua, pelo algoritmo de Prim.

        A distância de cada ponto só é calculada quando ele é visitado, o que mantém a memória em
        O(n) e o tempo em O(n^2). As primeiras `remaining` posições dos vetores guardam os pontos
        fora da árvore, de forma que cada passo só calcula as distâncias que ainda importam. Os
        quadrados das distâncias evitam a raiz quadrada a cada passo.
        """
        n = len(X)
        points = X.copy()
        ids = np.arange(n)
        squared_core = core_distances**2
        best = np.full(n, np.inf)
        attached_to = np.zeros(n, dtype=np.int64)
        edge_a = np.empty(n - 1, dtype=np.int64)
        edge_b = np.empty(n - 1, dtype=np.int64)
        edge_length = np.empty(n - 1)

        for remaining in range(n - 1, 0, -1):
            visited = ids[remaining]
            squared = cdist(points[remaining][None], points[:remaining], "sqeuclidean")[0]
            squared = np.maximum(squared, squared_core[ids[:remaining]])
            squared = np.maximum(squared, squared_core[visited])
            closer = squared < best[:remaining]
            best[:remaining][closer] = squared[closer]
            attached_to[:remaining][closer] = visited

            nearest = np.argmin(best[:remaining])
            last = remaining - 1
            for values in (points, ids, best, attached_to):
                values[[nearest, last]] = values[[last, nearest]]
            edge_a[last], edge_b[last], edge_length[last] = attached_to[last], ids[last], best[last]

        return edge_a, edge_b, np.sqrt(edge_length)

    def _build_merge_tree(self, edge_a, edge_b, edge_length):
        """
        Funde as arestas em ordem crescente e prepara as consultas por nó da árvore.

        Os nós 0 a n - 1 são os pontos, e cada fusão cria um nó de número maior que os dos filhos.
        Por isso, percorrer os números em ordem decrescente visita cada pai antes dos filhos.
        """
        n = self.n_points
        n_nodes = 2 * n - 1
        root = n_nodes - 1
        left = np.empty(n_nodes, dtype=np.int64)
        right = np.empty(n_nodes, dtype=np.int64)
        merge_height = np.zeros(n_nodes)
        self._subtree_size = np.ones(n_nodes, dtype=np.int64)
        component = list(range(n_nodes))

        def find(node):
            top = node
            while component[top] != top:
                top = component[top]
            while component[node] != top:
                component[node], node = top, component[node]
            return top

        order = np.argsort(edge_length, kind="stable")
        merges = zip(
            edge_a[order].tolist(), edge_b[order].tolist(), edge_length[order].tolist(), strict=True
        )
        for new_node, (a, b, length) in enumerate(merges, start=n):
            child_a, child_b = find(a), find(b)
            left[new_node], right[new_node], merge_height[new_node] = child_a, child_b, length
            self._subtree_size[new_node] = self._subtree_size[child_a] + self._subtree_size[child_b]
            component[child_a] = component[child_b] = new_node

        merged = np.arange(n, n_nodes)
        parent = np.arange(n_nodes)
        self._sibling = np.arange(n_nodes)
        self._parent_height = np.zeros(n_nodes)
        for child, other in ((left, right), (right, left)):
            parent[child[merged]] = merged
            self._sibling[child[merged]] = other[merged]
            self._parent_height[child[merged]] = merge_height[merged]

        # Os pontos de cada subárvore ocupam posições contíguas nesta ordem das folhas, o que
        # permite contar, com uma soma acumulada, quantos pontos de um conjunto há em cada nó.
        self._first_leaf = np.zeros(n_nodes, dtype=np.int64)
        first_leaf, sizes = self._first_leaf, self._subtree_size
        for node in range(root, n - 1, -1):
            first_leaf[left[node]] = first_leaf[node]
            first_leaf[right[node]] = first_leaf[node] + sizes[left[node]]

        # Cada tabela leva ao ancestral 2^j níveis acima, para acumular valores até a raiz em
        # O(log da altura) passos, mesmo em árvores muito desbalanceadas.
        self._ancestors = [parent]
        while (self._ancestors[-1] != root).any():
            self._ancestors.append(self._ancestors[-1][self._ancestors[-1]])

    def _fold_to_root(self, values, combine):
        """Combina os valores de cada nó com os de todos os seus ancestrais, até a raiz."""
        for ancestor in self._ancestors:
            values = combine(values, values[ancestor])
        return values[: self.n_points]

    def _count_below(self, mask):
        """Conta, em cada nó, os pontos marcados em mask."""
        marked = np.zeros(self.n_points + 1, dtype=np.int64)
        marked[self._first_leaf[: self.n_points][mask] + 1] = 1
        cumulative = np.cumsum(marked)
        return cumulative[self._first_leaf + self._subtree_size] - cumulative[self._first_leaf]

    def sum_distances_to(self, mask):
        """
        Soma, para cada ponto, as distâncias dc até os pontos marcados em mask.

        Objetivo:
            Cada ponto está a uma distância igual à altura de cada fusão de um ancestral dele de
            todos os pontos que entraram por meio do irmão nessa fusão. A soma percorre, então,
            esses ancestrais, com o peso de cada um dado pelos pontos marcados no irmão.

        Parâmetros:
            mask (np.ndarray): vetor booleano com n_points posições.

        Retorno:
            np.ndarray com a soma de cada ponto, sem contar a distância do ponto a si mesmo.

        Exemplo de uso:
            tree.sum_distances_to(labels == 0)  # soma até os pontos do agrupamento 0
        """
        counts = self._count_below(mask)
        return self._fold_to_root(self._parent_height * counts[self._sibling], np.add)

    def min_distance_to(self, mask):
        """
        Devolve, para cada ponto, a menor distância dc até os pontos marcados em mask.

        Objetivo:
            As alturas crescem em direção à raiz, de modo que a menor distância vem do ancestral
            mais baixo cujo irmão contém algum ponto marcado.

        Parâmetros:
            mask (np.ndarray): vetor booleano com n_points posições.

        Retorno:
            np.ndarray com a menor distância de cada ponto, sem contar o próprio ponto.

        Exemplo de uso:
            tree.min_distance_to(labels == -1)  # distância de cada ponto ao ruído mais próximo
        """
        counts = self._count_below(mask)
        heights = np.where(counts[self._sibling] > 0, self._parent_height, np.inf)
        heights[-1] = np.inf  # a raiz não tem pai
        return self._fold_to_root(heights, np.minimum)


def _relative_gap(larger_is_better, reference):
    """Calcula (a - b) / max(a, b), com valor 0 quando o denominador é zero (Beer et al., 2025)."""
    denominator = np.maximum(larger_is_better, reference)
    return np.divide(
        larger_is_better - reference,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )


def _cluster_point_scores(tree, labels, cluster_masks):
    """Calcula o escore dos pontos de agrupamento (Equações 4 e 8 de Beer et al., 2025)."""
    own_mean = np.zeros(tree.n_points)
    nearest_mean = np.full(tree.n_points, np.inf)
    for mask in cluster_masks:
        size = np.count_nonzero(mask)
        total = tree.sum_distances_to(mask)
        if size > 1:
            own_mean[mask] = total[mask] / (size - 1)
        nearest_mean[~mask] = np.minimum(nearest_mean[~mask], total[~mask] / size)

    # Com um único agrupamento, a separação é medida até o ruído mais próximo (Equação 8).
    if len(cluster_masks) == 1:
        nearest_mean = tree.min_distance_to(labels == NOISE_LABEL)

    scores = np.zeros(tree.n_points)
    clustered = labels != NOISE_LABEL
    scores[clustered] = _relative_gap(nearest_mean[clustered], own_mean[clustered])
    for mask in cluster_masks:
        if np.count_nonzero(mask) == 1:
            scores[mask] = 0.0  # agrupamento unitário (Seção 3.1)
    return scores


def _noise_point_scores(tree, noise, cluster_masks):
    """Calcula o escore dos pontos de ruído (Equações 5 a 7 de Beer et al., 2025)."""
    noise_core = tree.core_distances[noise]
    sparse = np.full(len(noise_core), np.inf)
    far = np.full(len(noise_core), np.inf)
    for mask in cluster_masks:
        cluster_core = np.full(len(noise_core), tree.core_distances[mask].max())
        sparse = np.minimum(sparse, _relative_gap(noise_core, cluster_core))
        far = np.minimum(far, _relative_gap(tree.min_distance_to(mask)[noise], cluster_core))
    return np.minimum(sparse, far)


def disco_samples(X, labels, min_points=DEFAULT_MIN_POINTS, tree=None):
    """
    Calcula o escore DISCO de cada ponto (BEER et al., 2025).

    Objetivo:
        O escore vai de -1 a 1, e valores maiores indicam um rótulo melhor. O ponto de agrupamento
        recebe a silhueta sobre a distância dc: compara a distância média ao próprio agrupamento
        com a distância média ao agrupamento mais próximo. O ponto de ruído é bem avaliado só se
        estiver em uma região esparsa e distante de todos os agrupamentos.

        Casos especiais definidos no artigo: sem agrupamentos, todos os pontos valem -1. Com um
        único agrupamento e sem ruído, ou em agrupamentos unitários, o escore é 0. Com um único
        agrupamento e ruído, a separação é medida até o ruído mais próximo.

    Parâmetros:
        X (np.ndarray): pontos, com forma (n_pontos, n_variáveis). Não é lido quando tree é dado,
            exceto para conferir o número de pontos.
        labels (np.ndarray): rótulo de cada ponto, com -1 para o ruído.
        min_points (int): parâmetro μ da distância de núcleo. O padrão é 5, como no artigo.
        tree (DCTree | None): árvore já construída sobre os mesmos X e min_points.

    Retorno:
        np.ndarray com o escore de cada ponto, na ordem de X.

    Exemplo de uso:
        scores = disco_samples(X, labels)
        scores[labels == -1].mean()   # qualidade dos rótulos de ruído
    """
    labels = np.asarray(labels).reshape(-1)
    if len(X) == 0:
        raise ValueError("Não há como calcular o DISCO de um conjunto vazio.")
    if len(X) != len(labels):
        raise ValueError(f"X tem {len(X)} pontos, e labels tem {len(labels)} rótulos.")
    if tree is not None and (tree.n_points != len(X) or tree.min_points != min_points):
        raise ValueError(
            f"A árvore foi construída para {tree.n_points} pontos e min_points={tree.min_points}, "
            f"e a chamada usa {len(X)} pontos e min_points={min_points}."
        )

    noise = labels == NOISE_LABEL
    cluster_masks = [labels == cluster for cluster in np.unique(labels[~noise])]
    if not cluster_masks:
        return np.full(len(labels), -1.0)
    if len(cluster_masks) == 1 and not noise.any():
        return np.zeros(len(labels))

    if tree is None:
        tree = DCTree(X, min_points)
    scores = _cluster_point_scores(tree, labels, cluster_masks)
    if noise.any():
        scores[noise] = _noise_point_scores(tree, noise, cluster_masks)
    return scores


def disco_score(X, labels, min_points=DEFAULT_MIN_POINTS, tree=None):
    """
    Calcula o DISCO de uma partição, a média dos escores de todos os pontos (BEER et al., 2025).

    Objetivo:
        Resume em um valor de -1 a 1 a qualidade da partição, inclusive dos rótulos de ruído, sem
        usar rótulos verdadeiros. É determinístico e está definido para qualquer partição.

    Parâmetros:
        X (np.ndarray): pontos, com forma (n_pontos, n_variáveis).
        labels (np.ndarray): rótulo de cada ponto, com -1 para o ruído.
        min_points (int): parâmetro μ da distância de núcleo. O padrão é 5, como no artigo.
        tree (DCTree | None): árvore já construída sobre os mesmos X e min_points, que evita
            repetir a parte mais cara ao avaliar várias partições do mesmo X.

    Retorno:
        float de -1 a 1, em que valores maiores indicam uma partição melhor.

    Exemplo de uso:
        disco_score(X_train, labels)

        tree = DCTree(X_train)
        for labels in partitions:
            disco_score(X_train, labels, tree=tree)
    """
    return float(np.mean(disco_samples(X, labels, min_points, tree)))
