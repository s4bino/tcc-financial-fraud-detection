> [!check] Status: em uso na monografia
> Todas as chaves constam do `refbib.bib` e são citadas em Metodologia › Detecção de Fraudes.
> - `fawcett2006`: classificação pelo escore e pelo limiar. Mesmo grifo da nota de métricas de avaliação.
> - `sklearn2026rf`: probabilidade de fraude do Random Forest e regra de decisão. É a documentação da biblioteca, sem PDF.
> - `xgboost2026`: saída do XGBoost como probabilidade. É a documentação da biblioteca, sem PDF.
> - `jain1999` apud `oyewole2023`: pertinência como grau de pertencimento ao agrupamento. O grifo de Jain et al. (1999) fica como conferência do original.
> - `cuml2026`: cálculo da pertinência no HDBSCAN\* e atribuição aproximada das transações novas. É a documentação da biblioteca, sem PDF.
> - `ramaswamy2000`: escore de anomalia kNN. Mesmo grifo da nota do HDBSCAN\*.
> - `dalpozzolo2018`: poucos alertas verificados por dia e preocupação com a precisão dos alertas.

## Escore e limiar

> [!PDF|yellow] [[fawcett2006_introduction-roc-analysis.pdf#page=3&selection=147,0,155,1&color=yellow|fawcett2006_introduction-roc-analysis, p.3]]
> > threshold to produce a discrete (binary) classifier: if the classifier output is above the threshold, the classifier produces a Y, else a N.
>
> Página 863 da revista. Sustenta que a transação é classificada como fraude quando o escore alcança o limiar. Mesmo grifo da nota de métricas de avaliação.

^fawcett2006-p3

## Modelos supervisionados

> [!quote] Documentação: [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), conferida em 23/10/2026
> > predict_proba: [...] mean predicted class probabilities of the trees in the forest. The class probability of a single tree is the fraction of samples of the same class in a leaf.
>
> Sustenta a probabilidade de fraude do Random Forest como a média das probabilidades das árvores.

^sklearn2026rf-doc

> [!quote] Documentação: [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), conferida em 23/10/2026
> > predict: [...] the predicted class is the one with highest mean probability estimate across the trees.
>
> Sustenta que a transação é classificada como fraude quando a probabilidade de fraude supera a de transação legítima.

^sklearn2026rf-doc-2

> [!quote] Documentação: [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), conferida em 23/10/2026
> > binary:logistic: logistic regression for binary classification, output probability
>
> Sustenta que a saída do XGBoost, com esse objetivo, é a própria probabilidade.

^xgboost2026-doc

## HDBSCAN*

> [!PDF|yellow] [[oyewole2023_data-clustering-application-trends.pdf#page=5&selection=32,0,33,75&color=yellow|oyewole2023_data-clustering-application-trends, p.5]]
> > This step is essentially described as the grouping process by Jain et al. (1999) into a partition of distinct groups or groups having a variable degree of membership.
>
> Página 6443 da revista, Seção 2.1.2. Sustenta, por apud de Jain et al. (1999), a pertinência como o grau com que um objeto pertence a um agrupamento.

^oyewole2023-p5

> [!PDF|yellow] [[jain1999_data-clustering-review.pdf#page=4&selection=73,0,75,21&color=yellow|jain1999_data-clustering-review, p.4]]
> > fuzzy (where each pattern has a variable degree of membership in each of the output clusters).
>
> Página 267 da revista. Trecho original a que Oyewole e Thopil (2023) se referem. A Seção 5.5, p. 281, define a pertinência como "the grade of membership of object xi in cluster cj" e afirma que "Larger membership values indicate higher confidence in the assignment of the pattern to the cluster".

^jain1999-p4

> [!quote] Documentação: [cuML API Reference](https://docs.nvidia.com/cuml/latest/api/), entrada `cuml.cluster.hdbscan.HDBSCAN`, conferida em 23/10/2026
> > probabilities_: The strength with which each sample is a member of its assigned cluster. Noise points have probability zero; points in clusters have values assigned proportional to the degree that they persist as part of the cluster.
>
> Sustenta que a probabilidade de pertinência é proporcional à persistência da transação no agrupamento e nula para o ruído.

^cuml2026-doc

> [!quote] Documentação: [cuML API Reference](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.hdbscan.approximate_predict/), entrada `cuml.cluster.hdbscan.approximate_predict`, conferida em 23/10/2026
> > Predict the cluster label of new points. The returned labels will be those of the original clustering found by clusterer, and therefore are not (necessarily) the cluster labels that would be found by clustering the original data combined with points_to_predict, hence the 'approximate' label. If you simply wish to assign new points to an existing clustering in the 'best' way possible, this is the function to use.
>
> Sustenta que a pertinência das transações da validação e do teste é estimada pela atribuição aproximada ao agrupamento do treino, sem reagrupar os dados.

^cuml2026-doc-2

> [!PDF|yellow] [[ramaswamy2000_mining-outliers-large-data-sets.pdf#page=1&selection=23,0,47,15&color=yellow|ramaswamy2000_mining-outliers-large-data-sets, p.1]]
> > In this paper, we propose a novel formulation for distance-based outliers that is based on the distance of a point from its kth nearest neighbor. We rank each point on the basis of its distance to its kth nearest neighbor and declare the top n points in this ranking to be outliers.
>
> Sustenta a distância ao $m_{pts}$-ésimo vizinho como escore de anomalia kNN. Mesmo grifo da nota do HDBSCAN\*.

^ramaswamy2000-p1

## Fração de alertas

> [!PDF|yellow] [[dalpozzolo2018_credit-card-fraud-realistic-modeling.pdf#page=2&selection=16,10,18,41&color=yellow|dalpozzolo2018_credit-card-fraud-realistic-modeling, p.2]]
> > these ignore the fact that only few alerts can be controlled everyday, and that companies are very concerned of the precision of the generated alerts.
>
> Página 3785 da revista. Sustenta a restrição dos sistemas reais: poucos alertas verificados por dia e preocupação com a precisão dos alertas.

^dalpozzolo2018-p2

