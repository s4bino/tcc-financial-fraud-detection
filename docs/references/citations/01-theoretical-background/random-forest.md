> [!check] Status: em uso na monografia
> - `biau2016`: consta do `refbib.bib`, citado em Fundamentação Teórica › Algoritmos Supervisionados › Random Forest.
> - `probst2019`: consta do `refbib.bib`, citado em Fundamentação Teórica › Algoritmos Supervisionados › Random Forest.

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=3|biau2016, p.3]]
> > Among the forests' essential ingredients, both bagging (Breiman 1996) and the Classification And Regression Trees (CART)-split criterion (Breiman et al. 1984) play critical roles. Bagging (a contraction of bootstrap-aggregating) is a general aggregation scheme, which generates bootstrap samples from the original data set, constructs a predictor from each sample, and decides by averaging. It is one of the most effective computationally intensive procedures to improve on unstable estimates, especially for large, high-dimensional data sets, where finding a good model in one step is impossible because of the complexity and scale of the problem (Bühlmann and Yu 2002; Kleiner et al. 2014; Wager et al. 2014).
>
> Página 199 na numeração da revista. Sustenta a instabilidade das estimativas de uma árvore isolada como a limitação que o RF supera.

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=16|biau2016, p.16]]
> > The idea of generating many bootstrap samples and averaging predictors is called bagging (bootstrap-aggregating). It was suggested by Breiman (1996) as a simple way to improve the performance of weak or unstable learners.
>
> Página 212 na numeração da revista. Reforça o trecho anterior: o *bagging* existe para melhorar aprendizes instáveis.

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=5|biau2016, p.5]]
> > Prior to the construction of each tree, $a_n$ observations are drawn at random with (or without) replacement from the original data set. [...] Then, at each cell of each tree, a split is performed by maximizing the CART-criterion (see below) over mtry directions chosen uniformly at random among the p original ones.
>
> Página 201 na numeração da revista. Sustenta o sorteio de um subconjunto de variáveis a cada divisão de nó.

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=9|biau2016, p.9]]
> > It is easy to see that the forest's variance decreases as M grows. Thus, more accurate predictions are likely to be obtained by choosing a large number of trees. Interestingly, picking a large M does not lead to overfitting.
>
> Página 205 na numeração da revista. Sustenta a redução da variância com o número de árvores, sem sobreajuste.

> [!PDF|yellow] [[biau2016_random-forest-guided-tour.pdf#page=20|biau2016, p.20]]
> > A seminal result by Breiman (2001) shows that the error of the forest is small as soon as the predictive power of each tree is good and the correlation between the tree errors is low. [...] Similarly, Friedman et al. (2009) decompose the variance of the forest as a product of the correlation between trees and the variance of a single tree.
>
> Página 216 na numeração da revista. Sustenta a variância da floresta como produto da correlação entre as árvores pela variância de uma árvore individual.
>
> *A random forest guided tour*
> Biau e Scornet

> [!PDF|yellow] [[probst2019_hyperparameters-tuning-random-forest.pdf#page=3|probst2019, p.3]]
> > Lower values of mtry lead to more different, less correlated trees, yielding better stability when aggregating.
>
> Sustenta que o sorteio de variáveis torna as árvores menos correlacionadas e a agregação mais estável.
>
> *Hyperparameters and tuning strategies for random forest*
> Probst, Wright e Boulesteix
