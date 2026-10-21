> [!check] Status: em uso na monografia
> Todas as chaves constam do `refbib.bib` e são citadas em Fundamentação Teórica › Algoritmo Não Supervisionado › HDBSCAN\*.
> - `campello2015`: base das definições da seção.
> - `campello2013b`: citado ao lado de `campello2015` na extração da partição.
> - `neto2022`: grafo conceitual e faixas de $m_{pts}$.
> - `neto2021`: citado ao lado de `neto2022` nas faixas de $m_{pts}$.
> - `ghosh2024`: ponto de partida da seção e fonte da premissa da detecção de anomalias e do efeito do $m_{pts}$.
> - `ramaswamy2000`: fonte original do escore kNN, citada diretamente.

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=2&selection=315,0,470,1&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.2]]
> > A. HDBSCAN* HDBSCAN* or Hierarchical DBSCAN* [5] is a hierarchical way of clustering data which is an improvement of one of the most popular density-based clustering algorithms, DBSCAN [16]. In DBSCAN*, density-based clusters are constructed using two parameters: (I) the radius ϵ, and (II) the minimum number of points minpts. A point xi in a dataset D is a core point if it has minpts data points in its ϵ–neighborhood Nϵ(xi) (i.e. in the set of points that are located less than or equal to distance ϵ away from xi); otherwise, a point is labeled as noise. Two core points are density connected w.r.t. ϵ and minpts if they directly or transitively belong to each other’s Nϵ. DBSCAN* defines a cluster as a maximal set of points where each pair of points in the set are density connected. HDBSCAN* improves DBSCAN* and does not require the user to pre-define an ϵ radius. For a minpts value, HDBSCAN* computes (I) the core distance ϵc(xi) for each xi ∈ D, which is the minimum ϵ distance required for xi to be a core point, and (II) the mutual reachability distance dmrd(xi, xj ), which is the minimum distance required for xi
>
> Ponto de partida da seção. As afirmações remetem às referências [5] (`campello2015`) e [16] (`ester1996`) do artigo.

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=3&selection=0,0,136,6&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.3]]
> > and xj to be in each other’s ϵ-neighborhood while both are also core points. The dmrd(xi, xj ) is defined as: dmrd(xi, xj ) = max{ϵc(xi), ϵc(xj ), d(xi, xj )} (1) where d(xi, xj ) represents the underlying distance between the two points in the dataset. The mutual reachability distance dmrd is dependent on the value of minpts. The HDBSCAN* hierarchy w.r.t. a single minpts value is obtained by computing the minimum spanning tree M STminpts of a complete, edgeweighted, virtual graph Gminpts [8], where the vertices represent the data points in the dataset, and the edge weights are the values of the mutual reachability distance dmrd between them. The edges from the MST are then removed in decreasing order of edge weights. At every removal, we obtain a new set of connected core objects (which are the clusters) and the remaining noise (points that are not connected). This creates a hierarchy of all possible DBSCAN* solutions obtained at ϵ ∈ [0, ∞] for a given minpts value.
>
> Continuação do trecho anterior. A árvore geradora mínima remete à referência [8] (`neto2022`).

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=7&selection=51,30,57,17&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.7]]
> > The core of the framework is a method based on nonparametric density estimates that gives rise to a hierarchical clustering algorithm HDBSCAN* (Hierarchical DBSCAN*). HDBSCAN* follows Hartigan’s model of density-contour clusters/trees and improves existing density-based clustering algorithms w.r.t. different aspects
>
> Página 6 na numeração da revista. Sustenta o HDBSCAN\* como aprimoramento dos algoritmos de agrupamento por densidade.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=10&selection=133,0,139,31&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.10]]
> > Our method has as its single input parameter a value for mpts. This is a classic smoothing factor in density estimates
>
> Página 9 na numeração da revista. Sustenta o $m_{pts}$ como único parâmetro de entrada e fator de suavização das estimativas de densidade.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=10&selection=79,0,99,37&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.10]]
> > Based on these definitions, we can devise an algorithm DBSCAN* (similar to DBSCAN) that conceptually finds clusters as the connected components of a graph in which the objects of X are vertices and every pair of vertices is adjacent if and only if the corresponding objects are ε-reachable w.r.t. user-defined parameters ε and mpts. Noncore objects are labeled as noise.
>
> Página 9 na numeração da revista. Sustenta que o DBSCAN\* exige a definição prévia do raio $\epsilon$, parâmetro definido pelo usuário.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=10&selection=164,0,243,16&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.10]]
> > Definition 3.1 (Core Distance). The core distance of an object xp ∈ X w.r.t. mpts, dcore(xp), is the distance from xp to its mpts-nearest neighbor (including xp). Notice that the core distance is the minimum radius ε such that xp satisfies the core condition w.r.t. mpts, that is, |Nε(xp)| ≥ mpts (Definition 2.1).
>
> Página 9 na numeração da revista. Sustenta a definição da distância núcleo e a sua leitura como menor raio que torna o ponto central.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=11&selection=4,0,120,38&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.11]]
> > Definition 3.2 (Mutual Reachability Distance). The mutual reachability distance between two objects xp and xq in X w.r.t. mpts is defined as dmreach(xp, xq) = max{dcore(xp), dcore(xq), d(xp, xq)}. Notice that dmreach(xp, xq) in Definition 3.2 is the minimum radius ε such that xp and xq are ε-reachable according to Definition 2.2.
>
> Página 10 na numeração da revista. Sustenta a Equação da distância de alcançabilidade mútua e a sua leitura como menor raio.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=11&selection=127,0,157,41&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.11]]
> > Definition 3.3 (Mutual Reachability Graph). The mutual reachability graph is a complete graph, Gmpts, in which the objects of X are vertices and the weight of each edge is the mutual reachability distance (w.r.t. mpts) between the respective pair of objects.
>
> Página 10 na numeração da revista. Sustenta o grafo de alcançabilidade mútua.

> [!PDF|yellow] [[neto2022_core-sg-multiple-msts-density-based.pdf#page=1&selection=196,0,209,13&color=yellow|neto2022_core-sg-multiple-msts-density-based, p.1]]
> > is typically obtained by dynamically computing edge weights for each pair of data points, representing adjacent vertices of a conceptual, complete graph of the data that does not need to be materialized.
>
> Página 951 dos anais. O sujeito da frase é a árvore geradora mínima (MST). Sustenta o grafo conceitual, com pesos calculados sob demanda.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=11&selection=181,31,216,1&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.11]]
> > it is straightforward to infer that clusters according to DBSCAN* w.r.t. mpts and ε are then the connected components of core objects in Gmpts,ε; and the remaining objects are noise. Consequently, all DBSCAN* clusterings for any ε ∈ [0, ∞) can be produced in a nested, hierarchical way by removing edges in decreasing order of weight from Gmpts
>
> Página 10 na numeração da revista. Sustenta a remoção das arestas em ordem decrescente de peso, com os componentes de pontos centrais como agrupamentos e o restante como ruído.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=12&selection=32,0,35,48&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.12]]
> > The fastest way to compute a Single-Linkage hierarchy is possibly by using a divisive algorithm based on the Minimum Spanning Tree (MST) [Jain and Dubes 1988], which works by removing edges from an MST in decreasing order of weights (here, corresponding to mutual reachability distances).
>
> Página 11 na numeração da revista. Sustenta a remoção das arestas da árvore geradora mínima.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=12&selection=77,81,92,38&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.12]]
> > The set of all unique clusterings that can be obtained by DBSCAN*, for a value of mpts and all possible values of the radius ε, corresponds one-to-one to the set of unique scale values (MSText edge weights) of the hierarchy levels.
>
> Página 11 na numeração da revista. Sustenta a hierarquia como o conjunto de todas as soluções do DBSCAN\* para o valor de $m_{pts}$.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=8&selection=15,53,17,89&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.8]]
> > we describe an algorithm that provides a globally optimal solution to the formal problem of maximizing the overall stability of the extracted clusters, for which a cluster stability measure is formulated.
>
> Página 7 na numeração da revista. Sustenta a extração da partição pelo critério de estabilidade.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=19&selection=127,3,136,51&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.19]]
> > Intuitively, when increasing λ (i.e., decreasing ε), clusters get smaller and smaller, until they disappear or break into subclusters. This observation gives rise to the intuition that, roughly speaking, more prominent clusters tend to “survive” longer after they appear
>
> Página 18 na numeração da revista. Sustenta que os agrupamentos mais proeminentes persistem por uma faixa maior de densidades.

> [!PDF|yellow] [[campello2013b_optimal-extraction-clusters-hierarchies.pdf#page=1&selection=24,0,28,34&color=yellow|campello2013b_optimal-extraction-clusters-hierarchies, p.1]]
> > We introduce a framework for the optimal extraction of flat clusterings from local cuts through cluster hierarchies. The extraction of a flat clustering from a cluster tree is formulated as an optimization problem and a linear complexity algorithm is presented that provides the globally optimal solution to this problem in semi-supervised as well as in unsupervised scenarios.
>
> Sustenta, ao lado de `campello2015`, a extração da partição a partir da hierarquia. É a referência [14] de Ghosh et al. para a extração automática de partições planas de uma hierarquia.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=15&selection=58,16,75,43&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.15]]
> > a connected component is deemed spurious if it has fewer than mclSize objects or, for mclSize = 1, if it is an isolated, non-dense object (a vertex with no edges). Any spurious component is labeled as noise.
>
> Página 14 na numeração da revista. Sustenta o tamanho mínimo de agrupamento, $m_{clSize}$, e o rótulo de ruído dos componentes espúrios.

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=3&selection=465,5,511,26&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.3]]
> > An increase in minpts values requires each point to have more points in its ϵ–neighborhood to become a core point, resulting in larger core distance ϵc(.) values. In turn, larger core distances tend to form “smoother” (and possibly larger) clusters in the hierarchy. Conversely, decreasing minpts values allows points to become core points with smaller ϵc(.) values, potentially building more detailed and smaller clusters within the base of the hierarchy.
>
> Sustenta o efeito do $m_{pts}$ sobre as distâncias núcleo e o tamanho dos agrupamentos. A frase anterior do artigo remete às referências [17] (`neto2021`) e [8] (`neto2022`).

> [!PDF|yellow] [[neto2022_core-sg-multiple-msts-density-based.pdf#page=1&selection=242,18,249,31&color=yellow|neto2022_core-sg-multiple-msts-density-based, p.1]]
> > The authors in [6], [19] have presented a collection of examples and visualizations that illustrate cases where different ranges of mpts values yield different results.
>
> Página 951 dos anais. Sustenta que faixas diferentes de $m_{pts}$ levam a resultados diferentes.

> [!PDF|yellow] [[neto2021_multiple-density-based-clustering-hierarchies.pdf#page=1&selection=19,2,33,76&color=yellow|neto2021_multiple-density-based-clustering-hierarchies, p.1]]
> > While a small change in mpts typically leads to a small change in the clustering structure, choosing a “good” mpts value can be challenging: depending on the data distribution, a high or low mpts value may be more appropriate, and certain clusters may reveal themselves at different values.
>
> Página 3075 da revista. Citado ao lado de `neto2022` para as faixas de $m_{pts}$. Pondera que uma mudança pequena no $m_{pts}$ costuma mudar pouco a estrutura, mas que a escolha do valor depende da distribuição dos dados.

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=2&selection=188,0,189,67&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.2]]
> > Unsupervised outlier detection is based on the premise that potential outliers lie in low-density regions of the reference set.
>
> Sustenta a premissa de que as anomalias estão em regiões de baixa densidade. É afirmação do próprio artigo, sem referência.

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=2&selection=203,23,230,10&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.2]]
> > A classic global method is k-Nearest Neighbor (kNN) [10], where the outlier score of a data point is the distance to its kth nearest neighbor (kNN–dist). In kNN the potential outliers are expected to have large
>
> Sustenta o escore kNN como método global clássico. Remete à referência [10] (`ramaswamy2000`).

> [!PDF|yellow] [[ramaswamy2000_mining-outliers-large-data-sets.pdf#page=1&selection=23,0,47,15&color=yellow|ramaswamy2000_mining-outliers-large-data-sets, p.1]]
> > In this paper, we propose a novel formulation for distance-based outliers that is based on the distance of a point from its kth nearest neighbor. We rank each point on the basis of its distance to its kth nearest neighbor and declare the top n points in this ranking to be outliers.
>
> Página 427 dos anais. Fonte original do escore kNN, citada diretamente no texto. Os símbolos $k$ e $n$ não existem na camada de texto do PDF, então o grifo cobre o trecho, mas a cópia sai sem eles.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=19&selection=72,34,107,1&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.19]]
> > DBSCAN* estimates density-contour clusters using a density threshold λ = 1/ε and a non-normalized K-NN estimate (for K = mpts) of f(x), given by 1/dcore(x)
>
> Página 18 na numeração da revista. Sustenta a analogia entre o escore kNN e a distância núcleo: o inverso da distância núcleo é uma estimativa de densidade por $K$ vizinhos, com $K = m_{pts}$.
