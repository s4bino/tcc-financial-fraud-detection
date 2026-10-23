> [!check] Status: em uso na monografia
> Todas as chaves constam do `refbib.bib` e são citadas em Metodologia › Parametrização e avaliação de consistência › Grades de Hiperparâmetros.
> - `probst2019`: compromisso entre correlação e qualidade das árvores, efeito de cada hiperparâmetro do Random Forest, valores padrão e tempo de execução. Só afirmações do próprio artigo, sem as que ele atribui a outros autores.
> - `biau2016`: mais árvores, predições mais precisas e sem sobreajuste. Mesmo trecho da nota do Random Forest.
> - `chen2004`: ponderação das classes no Random Forest e pesos como parâmetro de ajuste essencial.
> - `chen2016`: penalidade do número de folhas, regularização, *shrinkage* e configuração comum dos experimentos.
> - `xgboost2026`: peso das fraudes no XGBoost. É a documentação da biblioteca, sem PDF.
> - `sander1998` apud `schubert2017`: $m_{pts}$ igual ao dobro da dimensão e valores maiores para bases ruidosas, grandes ou de alta dimensão. Mesmo grifo da nota de seleção de hiperparâmetros.
> - `campello2015`: $m_{clSize}$ como controle adicional e independente da suavização.

## Random Forest

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=2&selection=87,0,91,28&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.2]]
> > In other words, an optimal compromise between low correlation and reasonable strength of the trees has to be found. This can be controlled by the parameters mtry, sample size, and node size
>
> Sustenta que os três primeiros hiperparâmetros da grade controlam o compromisso entre a baixa correlação das árvores e a qualidade de cada uma.

^probst2019-p2

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=8,0,17,77&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > One of the central hyperparameters of RF is mtry, as denoted in most RF packages, which is defined as the number of randomly drawn candidate variables out of which each split is selected when growing a tree. Lower values of mtry lead to more different, less correlated trees, yielding better stability when aggregating.
>
> Sustenta o número de variáveis sorteadas como hiperparâmetro central e o efeito dos valores menores.

^probst2019-p3

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=23,41,28,113&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > However, lower values of mtry also lead to trees that perform on average worse, since they are built based on suboptimal variables (that were selected out of a small set of randomly drawn candidates)
>
> Sustenta que cada árvore tende a ser pior quando escolhe as divisões entre poucas variáveis candidatas.

^probst2019-p3-2

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=2&selection=112,0,127,17&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.2]]
> > Typical default values mtry Number of drawn candidate variables in each split √p, p/3 for regression
>
> Tabela 1. Sustenta a raiz quadrada do número de variáveis como valor padrão na classificação. O PDF.js lê o símbolo da raiz como "ﬃﬃﬃ p".

^probst2019-p2-2

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=145,100,152,58&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > It has a similar effect as the mtry parameter. Decreasing the sample size leads to more diverse trees and thereby lower correlation between the trees, which has a positive effect on the prediction accuracy when aggregating the trees. However, the accuracy of the single trees decreases, since fewer observations are used for training.
>
> Sustenta o efeito do tamanho da amostra: árvores mais diversas, mas cada uma treinada com menos transações.

^probst2019-p3-3

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=161,33,161,80&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > Setting it to lower values reduces the runtime.
>
> Sustenta que amostras menores reduzem o tempo de execução.

^probst2019-p3-4

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=184,84,185,59&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > In several standard software packages the default value is 1 for classification and 5 for regression.
>
> Sustenta o valor padrão 1 do tamanho mínimo das folhas na classificação.

^probst2019-p3-5

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3&selection=189,15,190,63&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.3]]
> > In our experience, especially in large sample datasets it may be helpful to set this parameter to a value higher than the default one as it decreases the runtime substantially,
>
> Sustenta que folhas maiores reduzem o tempo de execução em bases grandes. A frase segue com uma referência a Segal (2004), que apoia só a ausência de perda de desempenho, não usada no texto.

^probst2019-p3-6

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=4&selection=129,0,129,65&color=yellow|probst2019_hyperparameters-tuning-random-forest, p.4]]
> > The computation time increases linearly with the number of trees.
>
> Sustenta o tempo de execução linear no número de árvores.

^probst2019-p4

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=9&selection=10,7,16,28&color=yellow|biau2016_random-forest-guided-tour, p.9]]
> > Thus, more accurate predictions are likely to be obtained by choosing a large number of trees. Interestingly, picking a large M does not lead to overfitting.
>
> Sustenta que mais árvores tendem a produzir predições mais precisas, sem sobreajuste.

^biau2016-p9

> [!PDF|yellow] [[chen2004_random-forest-imbalanced-data.pdf#page=3&selection=27,68,28,72&color=yellow|chen2004_random-forest-imbalanced-data, p.3]]
> > We assign a weight to each class, with the minority class given larger weight (i.e., higher misclassification cost).
>
> Seção 2.2. Sustenta a ponderação das classes, com peso maior para a classe minoritária.

^chen2004-p3

> [!PDF|yellow] [[chen2004_random-forest-imbalanced-data.pdf#page=3&selection=36,83,37,58&color=yellow|chen2004_random-forest-imbalanced-data, p.3]]
> > Class weights are an essential tuning parameter to achieve desired performance.
>
> Seção 2.2. Sustenta os pesos das classes como parâmetro de ajuste essencial.

^chen2004-p3-2

## XGBoost

> [!PDF|yellow] [[chen2016_xgboost-scalable-tree-boosting.pdf#page=3&selection=409,10,416,16&color=yellow|chen2016_xgboost-scalable-tree-boosting, p.3]]
> > Shrinkage scales newly added weights by a factor η after each step of tree boosting. Similar to a learning rate in tochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model.
>
> Seção 2.3. Sustenta a taxa de aprendizado como fator sobre a contribuição de cada nova árvore. A frase anterior atribui a técnica a Friedman, o que o texto não afirma.

^chen2016-p3

> [!PDF|yellow] [[chen2016_xgboost-scalable-tree-boosting.pdf#page=2&selection=197,0,200,5&color=yellow|chen2016_xgboost-scalable-tree-boosting, p.2]]
> > T is the number of leaves in the tree.
>
> Seção 2.1. Na Equação 2, a penalidade $\gamma T$ multiplica esse número de folhas. Sustenta que $\gamma$ penaliza o número de folhas.

^chen2016-p2

> [!PDF|yellow] [[chen2016_xgboost-scalable-tree-boosting.pdf#page=2&selection=305,18,308,12&color=yellow|chen2016_xgboost-scalable-tree-boosting, p.2]]
> > penalizes the complexity of the model (i.e., the regression tree functions). The additional regularization term helps to smooth the final learnt weights to avoid over-fitting.
>
> Seção 2.1. Sustenta o termo de regularização, que suaviza os pesos aprendidos e evita o sobreajuste.

^chen2016-p2-2

> [!PDF|yellow] [[chen2016_xgboost-scalable-tree-boosting.pdf#page=8&selection=195,27,197,20&color=yellow|chen2016_xgboost-scalable-tree-boosting, p.8]]
> > In all the experiments, we boost trees with a common setting of maximum depth equals 8, shrinkage equals 0.1
>
> Seção 6.2. Sustenta a configuração comum dos experimentos, com profundidade máxima 8 e taxa de aprendizado 0,1.

^chen2016-p8

> [!quote] Documentação: [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), conferida em 23/10/2026
> > scale_pos_weight [...] A typical value to consider: sum(negative instances) / sum(positive instances).
>
> Sustenta o peso das fraudes igual à razão entre transações legítimas e fraudes.

^xgboost2026-doc

## HDBSCAN*

> [!PDF|yellow] [[schubert2017_dbscan-revisited.pdf#page=11&selection=22,0,110,3&color=yellow|schubert2017, p.11]]
> > The easier-to-set parameter of DBSCAN is the minPts parameter. Its purpose is to smooth the density estimate, and for many datasets it can be kept at the default value of minPts = 4 (for two-dimensional data) [16]. Sander et al. [40] suggest setting it to twice the dataset dimensionality, i.e., minPts = 2 · dim. For datasets that have a lot of noise, that are very large, that are highdimensional, or that have many duplicates it may improve results to increase minPts. [...]
>
> Sustenta, por apud de Sander et al. (1998), o $m_{pts}$ igual ao dobro da dimensão e, como afirmação do próprio artigo, valores maiores para bases com muito ruído, muito grandes ou de alta dimensão. Mesmo grifo da nota de seleção de hiperparâmetros.

^schubert2017-p11

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=15&selection=79,0,86,25&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.15]]
> > Note that the optional use of the parameter mclSize represents an additional, independent control of the smoothing of the resulting cluster tree as a density estimate, in addition to the parameter
>
> Página 5:14 da revista, Seção 3.2. Sustenta o $m_{clSize}$ como controle adicional e independente da suavização. A frase termina com "mpts.".

^campello2015-p15

