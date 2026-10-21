> [!check] Status: em uso na monografia
> - `schubert2017`, `ester1996` e `campello2015`: constam do `refbib.bib`, citados em Fundamentação Teórica › Algoritmo Não Supervisionado › DBSCAN\*.

> [!PDF|yellow] [[schubert2017_dbscan-revisited.pdf#page=2&selection=51,0,93,33&color=yellow|schubert2017_dbscan-revisited, p.2]]
> > The model introduced by DBSCAN uses a simple minimum density level estimation, based on a threshold for the number of neighbors, minPts, within the radius ε (with an arbitrary distance measure). Objects with more than minPts neighbors within this radius (including the query point) are considered to be a core point. The intuition of DBSCAN is to find those areas, which satisfy this minimum density, and which are separated by areas of lower density. For efficiency reasons, DBSCAN does not perform density estimation in-between points. Instead, all neighbors within the ε radius of a core point are considered to be part of the same cluster as the core point (called direct density reachable). If any of these neighbors is again a core point, their neighborhoods are transitively included (density reachable). Non-core points in this set are called border points, and all points within the same set are density connected. Points which are not density reachable from any core point are considered noise and do not belong to any cluster.
>
> Sustenta a estimativa de níveis mínimos de densidade, os dois parâmetros, o ponto central e a premissa de regiões densas separadas por regiões mais esparsas, comuns ao DBSCAN e ao DBSCAN\*. O restante do trecho descreve o mecanismo dos pontos de borda, que o DBSCAN\* não tem.

> [!PDF|yellow] [[ester1996_density-based-algorithm-discovering-clusters.pdf#page=3&selection=438,0,451,41&color=yellow|ester1996_density-based-algorithm-discovering-clusters, p.3]]
> > In this section, we present the algorithm DBSCAN (Density Based Spatial Clustering of Applications with Noise) which is designed to discover the clusters and the noise in a spatial database according to definitions 5 and 6.
>
> Página 228 dos anais. Sustenta a expansão da sigla DBSCAN. Artigo original do algoritmo, referência [16] de Ghosh et al. e [Ester et al. 1996] de Campello et al. (2015).

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=9&selection=99,2,259,2&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.9]]
> > In the following, we define the density-based clustering algorithm DBSCAN* as in our preliminary work [Campello et al. 2013a], which differs from DBSCAN [Ester et al. 1996] in that the clusters are defined based on core objects alone. Definition 2.1 (Core and Noise Objects). An object xp is called a core object w.r.t. ε and mpts if its ε-neighborhood contains at least mpts many objects, that is, if |Nε(xp)| ≥ mpts, where Nε(xp) = {x ∈ X | d(x, xp) ≤ ε} and |·| denotes cardinality. An object is called noise if it is not a core object. Definition 2.2 (ε-Reachable). Two core objects xp and xq are ε-reachable w.r.t. ε and mpts if xp ∈ Nε(xq) and xq ∈ Nε(xp).
>
> Página 8 na numeração da revista. Sustenta o DBSCAN\* como variante do DBSCAN, com os agrupamentos formados apenas por pontos centrais, a definição de ponto central, o ruído como todo ponto não central e a alcançabilidade direta entre pontos centrais.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=10&selection=4,0,78,21&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.10]]
> > Definition 2.3 (Density-Connected). Two core objects xp and xq are density-connected w.r.t. ε and mpts if they are directly or transitively ε-reachable. Definition 2.4 (Cluster). A cluster C w.r.t. ε and mpts is a non-empty maximal subset of X such that every pair of objects in C is density-connected.
>
> Página 9 na numeração da revista. Sustenta a expansão transitiva do agrupamento pelos pontos centrais alcançáveis a partir dos seus membros.

> [!PDF|yellow] [[schubert2017_dbscan-revisited.pdf#page=3&selection=23,0,29,10&color=yellow|schubert2017_dbscan-revisited, p.3]]
> > In HDBSCAN* [13], the concept of border points was abandoned, and only core points are considered to be part of a cluster at any time, which is more consistent with the concepts of a density level set.
>
> Não citado no texto. Confirma que apenas pontos centrais compõem os agrupamentos, regra que o texto cita de `campello2015`.

> [!PDF|yellow] [[ankerst1999_optics-ordering-points-clustering-structure.pdf#page=1&selection=401,21,411,47&color=yellow|ankerst1999_optics-ordering-points-clustering-structure, p.1]]
> > Second, the algorithms are very sensible to these parameter values, often producing very different partitionings of the data set even for slightly different parameter settings.
>
> Página 49 da revista. Não citado no texto. Referência [12] de Ghosh et al. para a sensibilidade dos algoritmos de agrupamento aos parâmetros, o que motiva dispensar o raio $\epsilon$.
