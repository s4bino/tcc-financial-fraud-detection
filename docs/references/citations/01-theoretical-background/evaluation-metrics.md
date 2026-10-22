> [!check] Status: em uso na monografia
> Todas as chaves constam do `refbib.bib` e são citadas em Fundamentação Teórica › Métricas de Avaliação.
> - `fawcett2006`: matriz de confusão, acurácia, precisão, revocação, F1 e o escore com limiar.
> - `chawla2002`: acurácia inadequada em bases desbalanceadas. Citado pelo Alcino (2022) para a mesma afirmação.
> - `davis2006`: curva de precisão e revocação, comparação com a ROC e erro da interpolação linear. Citado por Jurgovsky et al. (2018) para a escolha da AUPRC.
> - `campos2016`: precisão média na avaliação de detectores de anomalias. A equação remete a `zhang2009`.
> - `zhang2009`: fonte original da equação da precisão média.
> - `moulavi2014`: critérios internos de validação e o índice DBCV. Também citado no Capítulo 2.

## Matriz de confusão e métricas derivadas

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=2&selection=7,35,31,20&color=yellow|fawcett2006_introduction-roc-analysis, p.2]]
> > there are four possible outcomes. If the instance is positive and it is classified as positive, it is counted as a true positive; if it is classified as negative, it is counted as a false negative. If the instance is negative and it is classified as negative, it is counted as a true negative; if it is classified as positive, it is counted as a false positive. Given a classifier and a set of instances (the test set), a two-by-two confusion matrix (also called a contingency table) can be constructed representing the dispositions of the set of instances. This matrix forms the basis for many common metrics.
>
> Página 862 da revista. Sustenta a matriz de confusão e a definição das quatro contagens.

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=2&selection=222,0,222,75&color=yellow|fawcett2006_introduction-roc-analysis, p.2]]
> > Fig. 1. Confusion matrix and common performance metrics calculated from it.
>
> Página 862 da revista. A Figura 1 traz a disposição da matriz, adaptada no Quadro 3.1, e as fórmulas de acurácia, precisão, revocação e F-measure (Equações 3.2 a 3.5). As fórmulas estão desenhadas na figura, fora da camada de texto, por isso o grifo fica na legenda.

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=2&selection=37,0,50,25&color=yellow|fawcett2006_introduction-roc-analysis, p.2]]
> > The true positive rate (also called hit rate and recall) of a classifier is estimated as
>
> Página 862 da revista. Sustenta que a revocação é a taxa de verdadeiros positivos.

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=2&selection=75,0,79,6&color=yellow|fawcett2006_introduction-roc-analysis, p.2]]
> > sensitivity = recall
>
> Página 862 da revista. Sustenta que a revocação também é chamada de sensibilidade.

> [!PDF|yellow] [[chawla2002_smote-synthetic-minority-oversampling.pdf#page=2&selection=11,0,17,14&color=yellow|chawla2002_smote-synthetic-minority-oversampling, p.2]]
> > The performance of machine learning algorithms is typically evaluated using predictive accuracy. However, this is not appropriate when the data is imbalanced and/or the costs of different errors vary markedly. As an example, consider the classification of pixels in mammogram images as possibly cancerous (Woods, Doss, Bowyer, Solka, Priebe, & Kegelmeyer, 1993). A typical mammography dataset might contain 98% normal pixels and 2% abnormal pixels. A simple default strategy of guessing the majority class would give a predictive accuracy of 98%.
>
> Página 322 da revista. Sustenta que a acurácia não é adequada em bases desbalanceadas e o exemplo do classificador que atribui tudo à classe majoritária.

## Escore, limiar e curva de precisão e revocação

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=3&selection=147,0,155,1&color=yellow|fawcett2006_introduction-roc-analysis, p.3]]
> > threshold to produce a discrete (binary) classifier: if the classifier output is above the threshold, the classifier produces a Y, else a N.
>
> Página 863 da revista. Sustenta a classificação pelo escore comparado a um limiar.

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=2&selection=125,0,127,22&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.2]]
> > A confusion matrix is shown in Figure 2(a). The confusion matrix can be used to construct a point in either ROC space or PR space.
>
> Página 234 dos anais. Com o trecho seguinte, sustenta que cada limiar produz um par de precisão e revocação, isto é, um ponto da curva.

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=5&selection=245,0,247,15&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.5]]
> > Thus each point in ROC space or PR space represents a specific classifier, with a threshold for calling an example positive.
>
> Página 237 dos anais. Cada ponto da curva corresponde a um limiar.

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=1&selection=88,0,90,54&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.1]]
> > However, ROC curves can present an overly optimistic view of an algorithm's performance if there is a large skew
>
> Página 233 dos anais. Sustenta a visão otimista da ROC com grande desbalanceamento.

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=1&selection=108,0,112,9&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.1]]
> > Precision-Recall (PR) curves, often used in Information Retrieval (Manning & Schutze, 1999; Raghavan et al., 1989), have been cited as an alternative to ROC curves for tasks with a large skew in the class distribution
>
> Página 233 dos anais. Sustenta a curva de precisão e revocação como alternativa.

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=1&selection=149,14,155,31&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.1]]
> > a large change in the number of false positives can lead to a small change in the false positive rate used in ROC analysis. Precision, on the other hand, by comparing false positives to true positives rather than true negatives, captures the effect of the large number of negative examples on the algorithm
>
> Página 233 dos anais. Sustenta o efeito dos falsos positivos na taxa de falsos positivos e na precisão.

## Precisão média

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=6&selection=32,22,34,22&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.6]]
> > In these cases, linear interpolation is a mistake that yields an overly-optimistic estimate of performance.
>
> Página 238 dos anais. Sustenta que a interpolação linear entre os pontos da curva é incorreta e superestima o desempenho.

> [!PDF|yellow] [[campos2016_evaluation-unsupervised-outlier-detection.pdf#page=9&selection=124,0,125,11&color=yellow|campos2016_evaluation-unsupervised-outlier-detection, p.9]]
> > The methods studied here return a full ranking of database objects based on their outlierness
>
> Início da Seção 3 (Evaluation measures). Sustenta que os métodos de detecção de anomalias produzem uma ordenação pelo grau de anomalia.

> [!PDF|yellow] [[campos2016_evaluation-unsupervised-outlier-detection.pdf#page=11&selection=44,2,45,82&color=yellow|campos2016_evaluation-unsupervised-outlier-detection, p.11]]
> > One such measure, popular in information retrieval contexts, is the average precision (Zhang and Zhang 2009)
>
> Sustenta a precisão média como medida de avaliação da ordenação. A equação (Eq. 14 do artigo) remete a Zhang e Zhang (2009), citados na Equação 3.6.

> [!PDF|yellow] [[zhang2009_average-precision.pdf#page=192&selection=186,0,226,20&color=yellow|zhang2009_average-precision, p.192]]
> > Average precision is a measure that combines recall and precision for ranked retrieval results. For one information need, the average precision is the mean of the precision scores after each relevant document is retrieved. Average Precision = Σr P@r / R where r is the rank of each relevant document, R is the total number of relevant documents, and P@r is the precision of the top-r retrieved documents.
>
> Página 192 da enciclopédia (o PDF reúne todos os verbetes da letra A). Sustenta a Equação 3.6: a média das precisões nas posições dos exemplos positivos. Na monografia, R é |O| e r é pos(o).

> [!PDF|yellow] [[campos2016_evaluation-unsupervised-outlier-detection.pdf#page=11&selection=68,0,87,73&color=yellow|campos2016_evaluation-unsupervised-outlier-detection, p.11]]
> > Instead of evaluating only at a single value of n, the values of P@n are averaged over the ranks of all outlier objects o ∈ |O|. There exist other variants, such as evaluating over a sample of the ranks at fixed intervals. However, evaluation at positive (outlier) ranks appears to be both popular and well-suited for imbalanced datasets.
>
> Sustenta que avaliar a precisão nas posições dos positivos é adequado para bases desbalanceadas.

> [!PDF|yellow] [[campos2016_evaluation-unsupervised-outlier-detection.pdf#page=11&selection=93,2,94,67&color=yellow|campos2016_evaluation-unsupervised-outlier-detection, p.11]]
> > A perfect ranking yields a maximum value of 1, and the expected value of a random ranking is
>
> Sustenta o valor 1 na ordenação perfeita e o valor esperado da ordenação aleatória, que o artigo completa com |O|/N, a proporção de positivos.

## Índice DBCV

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=1&selection=119,0,121,28&color=yellow|moulavi2014_density-based-clustering-validation, p.1]]
> > Because internal validity criteria measure clustering quality based solely on information intrinsic to the data
>
> Sustenta os critérios internos, que usam apenas os próprios dados.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=1&selection=125,0,134,53&color=yellow|moulavi2014_density-based-clustering-validation, p.1]]
> > The vast majority of relative validity criteria are based on the idea of computing the ratio of within-cluster scattering (compactness) to between-cluster separation. Measures that follow this definition have been designed for the evaluation of convex shaped clusters (e.g., globular clusters) and fail when applied to validate arbitrarily shaped, non-convex clusters. They are also not defined for noise objects.
>
> Sustenta a limitação dos critérios tradicionais: formato convexo e ausência de definição para o ruído.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=1&selection=146,16,147,41&color=yellow|moulavi2014_density-based-clustering-validation, p.1]]
> > we propose a measure called the Density-Based Clustering Validation index (DBCV).
>
> Sustenta a proposição do DBCV.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=3&selection=87,0,90,46&color=yellow|moulavi2014_density-based-clustering-validation, p.3]]
> > Considering such a model we can expect a good density-based clustering solution to have clusters in which the lowest density area inside each cluster is still denser than the highest density area surrounding clusters.
>
> Sustenta a ideia de que parte o índice.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=3&selection=105,30,123,31&color=yellow|moulavi2014_density-based-clustering-validation, p.3]]
> > To formulate DBCV we define the notion of all-points-core-distance (apts coredist) which is the inverse of the density of each object with respect to all other objects inside its cluster. Using apts coredist we define a symmetric reachability distance (similar to the definition by Lelis and Sander [22]) which is then employed to build a Minimum Spanning Tree (MST) inside each cluster.
>
> Sustenta a distância núcleo calculada com todos os objetos do grupo, como inverso da densidade, e a árvore geradora mínima de cada grupo.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=4&selection=342,0,342,44&color=yellow|moulavi2014_density-based-clustering-validation, p.4]]
> > Definition 2. (Mutual Reachability Distance)
>
> A definição, logo abaixo do título, é a mesma da Equação 3.1 com a distância núcleo do DBCV no lugar da distância núcleo do HDBSCAN\*. O texto da definição está intercalado com a outra coluna na camada de texto, por isso o grifo fica no título.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=4&selection=560,0,561,56&color=yellow|moulavi2014_density-based-clustering-validation, p.4]]
> > Let the set of internal edges in the MST be all edges except those with one ending vertex of degree one.
>
> Sustenta a definição de arestas internas.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=0,0,1,29&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Let the set of internal objects (vertices) be all objects except those with degree one.
>
> Sustenta a definição de objetos internos.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=6,0,18,53&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Definition 5. (Density Sparseness of a Cluster) The Density Sparseness of a Cluster (DSC) Ci is defined as the maximum edge weight of the internal edges
>
> Sustenta a esparsidade de densidade do grupo. O trecho anterior do artigo a interpreta como a região menos densa do grupo.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=49,0,83,55&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Definition 6. (Density Separation) The Density Separation of a Pair of Clusters (DSPC) Ci and Cj, 1 ≤ i, j ≤ l, i ≠ j, is defined as the minimum reachability distance between the internal nodes of the
>
> Sustenta a separação de densidade entre grupos, entre os objetos internos das árvores.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=110,0,112,34&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Definition 7. (Validity Index of a Cluster) We define the validity of a cluster
>
> Sustenta a Equação 3.7, logo abaixo no artigo (Eq. 3.4).

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=102,33,108,56&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Note that, if a cluster has better density compactness than density separation we obtain positive values of the validity index. If the density inside a cluster is lower than the density that separates it from other clusters, the index is negative.
>
> Sustenta a leitura do sinal da validade do grupo.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=232,0,232,46&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Definition 8. (Validity Index of a Clustering)
>
> Sustenta a Equação 3.8, a média das validades ponderada pelo tamanho dos grupos (Eq. 3.5 do artigo).

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=215,0,230,13&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > Note that, although noise is not explicitly present in our formulation, it is implicitly considered by the weighted average that takes into account the size of the cluster (|Ci|) and the total number of objects under evaluation, including noise, given by |O| in Eq. (3.5).
>
> Sustenta que |O| inclui o ruído.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=5&selection=315,0,325,53&color=yellow|moulavi2014_density-based-clustering-validation, p.5]]
> > It is easy to verify that our index produces values between −1 and +1, with greater values of the measure indicating better density-based clustering solutions.
>
> Sustenta o intervalo do índice e a sua leitura.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=6&selection=11,0,15,50&color=yellow|moulavi2014_density-based-clustering-validation, p.6]]
> > Just removing the noise without any penalty in approach (iv) is not a good strategy because the coverage is not considered. For instance, a solution which has one object from each cluster and all other objects as noise results in a perfect score.
>
> Com o trecho seguinte, sustenta que o ruído no denominador penaliza as soluções que deixam muitos objetos fora dos grupos.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=6&selection=39,0,42,5&color=yellow|moulavi2014_density-based-clustering-validation, p.6]]
> > Note that this is the same approach adopted for DBCV (Eq. 3.5).
>
> Liga a penalidade pela cobertura ao próprio DBCV.

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=1&selection=42,45,45,51&color=yellow|moulavi2014_density-based-clustering-validation, p.1]]
> > Experiments on synthetic and real world data show the effectiveness of our approach for the evaluation and selection of clustering algorithms and their respective appropriate parameters.
>
> Sustenta o uso do DBCV na avaliação e na seleção de algoritmos e parâmetros.
