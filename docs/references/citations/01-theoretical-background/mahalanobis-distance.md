> [!todo] Status: leitura de estudo, ainda sem texto na monografia
> Trechos que sustentam o estudo da distância de Mahalanobis (`docs/scratch/notes/mahalanobis-distance-study.md`), feito antes de levá-la ao pipeline do HDBSCAN\*. Nenhuma das chaves novas consta do `refbib.bib`.
> - `ghorbani2019`: definição, relação com a euclidiana, mascaramento e versão robusta pelo MCD.
> - `demaesschalck2000`: distâncias euclidiana e de Mahalanobis no espaço original e no dos componentes principais.
> - `todeschini2013`: forma quadrática geral, invariância à escala, mascaramento e encobrimento, e os dois tipos de anomalia (remota e isolada).
> - `evans2001`: capítulo 14 do livro organizado por Lock, Wells e Farthing. O PDF é o livro inteiro, e só o trecho sobre Mahalanobis foi lido.
> - `xie2024`: Mahalanobis na mesma base de cartões, com as ressalvas metodológicas anotadas abaixo.
> - `laca`: sem PDF. O resumo e o começo da introdução estão em `laca_runtime-optimization-mahalanobis-credit-card-fraud.txt`, e o ano e o veículo não foram localizados.
> - `campello2015`: já consta do `refbib.bib`. Os trechos abaixo são novos e não estão citados no `main.tex`.

## Definição e relação com a distância euclidiana

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=2&selection=1515,0,1631,3&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.2]]
> > It can be easily noted that formula (1) is a squared Euclidean distance if M = I, where I is the identity matrix, and a weighted Euclidean distance if M = W, where W is a symmetric weight matrix. Moreover, if M = Σ⁻¹ where Σ is the population covariance matrix, the squared Mahalanobis distance is obtained as:
>
> Sustenta as três distâncias como casos da mesma forma quadrática, $(x - \mu)^T M (x - \mu)$, que muda só pela matriz $M$.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=6&selection=105,0,106,26&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.6]]
> > If the covariance matrix is the identity matrix, the Mahalanobis distance reduces to the Euclidean distance.
>
> Página 588 da revista. Sustenta a euclidiana como caso particular da Mahalanobis.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=6&selection=27,63,29,30&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.6]]
> > It differs from Euclidean distance in this way that it takes into account the correlations between variables. It is a scale invariant metric
>
> Página 588 da revista. Sustenta que a Mahalanobis considera as correlações e não depende da escala das variáveis.

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=3&selection=535,0,575,8&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.3]]
> > Like the distances derived using the classical covariance matrix, the locally centred Mahalanobis distances are invariant to any sort of variable scaling.
>
> Sustenta a invariância à escala da Mahalanobis clássica. Por ela, a padronização prévia de `log_amount` não altera a distância.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=6&selection=15,45,20,70&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.6]]
> > One problem with the Euclidean distance measure is that it does not take the correlation between highly correlated variables into account. In this situation, Euclidean distance assigns equal weight to such variables, and since these variables measure essentially the same characteristic, therefore this single characteristic gets additional weight. In effect, correlated variables gets excess weight by Euclidean distance, see [16] and [21].
>
> Página 588 da revista. Sustenta o peso excessivo que a euclidiana dá a variáveis correlacionadas. Remete às referências [16] (Jolliffe, 1986) e [21] (Mimmack et al., 2001), fora do acervo.

> [!PDF|yellow] [[xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud.pdf#page=3&selection=147,28,150,61&color=yellow|xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud, p.3]]
> > Compared to the Euclidean distance, the Mahalanobis distance considers the correlation between features and, therefore, can more accurately describe the distance between samples when feature correlation exists.
>
> Página 162790 da revista. Sustenta a mesma comparação na base de cartões usada no trabalho.

> [!PDF|yellow] [[xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud.pdf#page=3&selection=285,0,289,33&color=yellow|xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud, p.3]]
> > Since the Euclidean distance uses the same weight for all features when calculating the distance, it may produce some more key features having less impact on the distance calculation results, and the Euclidean distance does not consider the correlation between features.
>
> Página 162790 da revista. Sustenta que o peso igual da euclidiana pode reduzir a influência das variáveis mais informativas.

> [!PDF|yellow] [[evans2001_statistical-aspects-detection-fraud.pdf#page=197&selection=45,0,66,24&color=yellow|evans2001_statistical-aspects-detection-fraud, p.197]]
> > The distance of a value from the mean for one variable can be expressed in units of standard deviation, a "Z score". This distance can be generalised to two dimensions when the distance from a bivariate mean (for example, diastolic blood pressure and sodium concentration each expressed as Z scores) can be calculated using Pythagoras' theorem. A measure of the distance from a multivariate mean, equivalent to the square of a Z score is called the Mahalanobis distance. The distribution of this distance should follow a chi-square distribution, approximately. Very large values-outliers can be detected in this way.
>
> Sustenta a Mahalanobis como generalização multivariada do escore Z e seu uso na detecção de valores extremos, no contexto da fraude em dados de pesquisa.

## Distâncias no espaço dos componentes principais

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=2&selection=78,35,86,87&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.2]]
> > In this case, the MD, however, does not need to correct for the covariance between the variables, since PCs are by definition orthogonal (uncorrelated). However, the way each of the residual PCs is weighted in the computation of the distance must be taken into account.
>
> Sustenta que, sobre componentes principais como V1 a V28, a Mahalanobis clássica não tem correlação a corrigir e só muda o peso de cada componente.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=7&selection=97,41,103,28&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.7]]
> > This is because the ED computed in the unnormalised PC space does not give equal importance to each of the PCs, but weighs each PC according to the amount of the total variance in the data explained by it.
>
> Sustenta que a euclidiana sobre os escores da PCA pesa cada componente pela variância que ele explica, como acontece hoje com V1 a V28 sem padronização.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=7&selection=90,14,91,106&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.7]]
> > As will be shown in the next paragraph, the ED computed in the normalised PC space is, except for a constant factor, equal to the MD computed in that space.
>
> Sustenta que padronizar os componentes e usar a euclidiana equivale à Mahalanobis clássica, a menos de um fator constante.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=11&selection=649,34,652,44&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.11]]
> > It can be seen that in the original variable space, the ED and MD are indeed clearly different. The MDs in all three spaces are the same when using all PCs in the computations. The EDs in the original space and the PC space are also the same when using all PCs. The ED in the normalised PC space is equal to the MDs, but for a constant factor.
>
> Sustenta as equivalências entre as distâncias no espaço original e no dos componentes (Equações 20, 22, 24 e 25 do artigo), quando todos os componentes são usados.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=14&selection=84,103,87,44&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.14]]
> > In this application of the MD, it is an advantage that the important PCs are equally weighted, since one wants to see all changes that occur in the measurements even when it occurs in a PC that explains a small amount of the total variance within the training set data.
>
> Sustenta o argumento a favor do peso igual dos componentes no controle de processos: um desvio num componente de pouca variância também é anomalia.

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=7&selection=289,0,359,9&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.7]]
> > It is noteworthy that, unlike Euclidean distance, both the locally centred and classical Mahalanobis distances do not highlight samples 263 and 317 as apparent remote sample since they are located along the directions of maximum data variance.
>
> Sustenta que a euclidiana aponta como remotos pontos que só se afastam na direção de maior variância.

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=7&selection=414,0,520,7&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.7]]
> > A deeper analysis of the 8 original variables for the Italoil data set did not provide any anomalous value for the sample 390; however, through PCA, it was possible to find out that considering the last two PCs (7 and 8, explained variance 1.52%, Fig. 8) the sample 390 is completely far from the data domain.
>
> Sustenta que uma anomalia pode aparecer só nos componentes de menor variância, onde a Mahalanobis a detecta.

## Sensibilidade às anomalias: mascaramento e encobrimento

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=5&selection=58,0,77,74&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.5]]
> > A problem with parametric approaches is that distribution parameters such as mean, standard deviation, and covariances are rather sensitive to the presence of outliers. Possible effects of outliers on the parameter estimation have been termed "masking" and "swamping". Outliers can mask their own presence by influencing the values of the distribution parameters (resulting in false negatives), or swamp inliers to appear as outlying due to the influenced parameters (resulting in false positives)
>
> Página 5:4 da revista. Sustenta, na obra do HDBSCAN\* já citada no trabalho, que a covariância é sensível às anomalias, e define mascaramento e encobrimento.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=2&selection=23,57,31,9&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.2]]
> > Due to the masking effect, in which one outlier masks a second outlier, if the second outlier can be considered as an outlier only by itself, but not in the presence of the first outlier, detecting multiple outliers is more completed than the case where data consist of a single outlier, since masking effects might decrease the Mahalanobis distance of an outlier. This might happen because a small cluster of outliers attracts mean and inflate variance towards its direction [4]. In such cases using robust estimates of sample mean and variance, can often improve the performance of the detection procedure, see [24] and [30].
>
> Página 584 da revista. Sustenta que um pequeno grupo de anomalias atrai a média e infla a variância na sua direção, o que reduz a Mahalanobis delas. Remete às referências [4] (Becker e Gather, 1999), [24] (Rocke e Woodruff, 1996) e [30] (Santos-Pereira e Pires, 2002), fora do acervo.

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=2&selection=786,0,896,9&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.2]]
> > Many times the presence of some outliers can somehow mask the detection of other outliers. As a result, some outliers are wrongly identified as normal samples. This phenomenon is referred to as masking. On the contrary, swamping refers to the cases where the presence of a subset of observations makes normal samples being incorrectly identified as potential outliers.
>
> Sustenta as definições de mascaramento e encobrimento.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=12&selection=238,30,247,51&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.12]]
> > Indeed, if an outlier is present, the estimated mean x̄ and variance–covariance matrix Cx for computing the MDs are already influenced by it.
>
> Sustenta que a média e a covariância estimadas já carregam a influência da anomalia que se quer detectar.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=8&selection=136,20,169,30&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.8]]
> > The classical approach of outlier detection uses the estimates of the Mahalanobis distance, by plugging in multivariate sample mean X̄ and covariance matrix S estimates for unknown mean µ and covariance matrix Σ, and tags as outlier any observation which has a Mahalanobis squared distance d²(X, X̄) lying above a predefined quantile of the χ² distribution with p degrees of freedom [7].
>
> Página 590 da revista. Sustenta a regra clássica de corte pelo quantil da qui-quadrado, que o pipeline não usa: o limiar sai do quantil dos escores do treino.

## Estimadores robustos

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=8&selection=170,0,175,56&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.8]]
> > This method is problematic, because all relies on normality assumption and the parameters estimates are particularly sensitive to outliers. Therefore, it is important to consider robust alternatives to these estimators for calculating robust Mahalanobis distances. The most widely used estimator of this type is the mini-
>
> Página 590 da revista. Sustenta a dependência da normalidade e a sensibilidade das estimativas clássicas, e a necessidade de estimadores robustos. Continua no trecho seguinte.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=9&selection=3,0,4,41&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.9]]
> > mum covariance determinant (MCD) estimator defined in [25] for which also a fast computing algorithm was constructed [27].
>
> Página 591 da revista. Sustenta o MCD como o estimador robusto mais usado. Remete às referências [25] (Rousseeuw, 1985) e [27] (Rousseeuw e Van Driessen, 1999), fora do acervo. A segunda é o algoritmo FastMCD, implementado no `MinCovDet` do scikit-learn.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=13&selection=132,44,163,66&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.13]]
> > Rousseeuw [31,32] introduced a robust version of the MD as a tool for the detection of outliers, the minimum volume ellipsoid (MVE) estimator. The variance–covariance is estimated using the at least (n/2 + 1) objects of X which cover the smallest possible ellipsoid. In this way, the variance–covariance matrix is not influenced by multiple outliers.
>
> Sustenta a ideia comum aos estimadores robustos: estimar a covariância só com o núcleo dos dados. Remete às referências [31] (Rousseeuw, 1983) e [32] (Rousseeuw, 1997), fora do acervo.

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=10&selection=35,0,36,52&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.10]]
> > In other words, applying the robust method enabled us to detect hidden outliers which has been masked by each other.
>
> Página 592 da revista. Sustenta que a versão robusta revela anomalias mascaradas umas pelas outras.

## Anomalias remotas, isoladas e internas

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=3&selection=1891,0,2015,5&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.3]]
> > The remoteness highlights objects which are far from the bulk of the remaining objects, i.e. they can be considered as classical outliers in the selected variable space; the Isolation degree detects a different kind of "anomalous" objects, i.e. those objects that, although located within the variable space, are isolated from the other ones or, in other words, these objects are surrounded by objects not so near.
>
> Sustenta a distinção entre anomalia remota, longe do grosso dos dados, e anomalia isolada, dentro do domínio mas sem vizinhos próximos.

> [!PDF|yellow] [[evans2001_statistical-aspects-detection-fraud.pdf#page=196&selection=86,0,89,9&color=yellow|evans2001_statistical-aspects-detection-fraud, p.196]]
> > The problem with an invented data point is that it is unlikely to be an outlier in any dimension; in fact, the exact opposite is true. Invented data are likely to have values that lie close to the mean for each variable that has been measured.
>
> Sustenta que uma fraude que imita o comportamento típico fica perto da média e escapa de qualquer detector de valores extremos.

> [!PDF|yellow] [[evans2001_statistical-aspects-detection-fraud.pdf#page=197&selection=66,25,67,75&color=yellow|evans2001_statistical-aspects-detection-fraud, p.197]]
> > Although not mentioned in textbooks, this can also be used to detect "inliers", looking for much smaller values of Mahalanobis
>
> Sustenta o uso da Mahalanobis também para pontos próximos demais do centro. O trecho continua com "distance." na p. 198.

## Custo e estabilidade numérica

> [!PDF|yellow] [[ghorbani2019_mahalanobis-distance-multivariate-outliers.pdf#page=2&selection=17,0,20,10&color=yellow|ghorbani2019_mahalanobis-distance-multivariate-outliers, p.2]]
> > For large data set using the Mahalanobis distance is computationally more expensive than Euclidean distance as it require to pass through all variables in data set to calculate the underlying inter-correlation structure.
>
> Página 584 da revista. Sustenta o custo extra da Mahalanobis em bases grandes. No pipeline, esse custo fica numa transformação linear feita uma vez por fold.

> [!PDF|yellow] [[demaesschalck2000_mahalanobis-distance-tutorial.pdf#page=2&selection=52,79,54,7&color=yellow|demaesschalck2000_mahalanobis-distance-tutorial, p.2]]
> > This so-called multicollinearity in the data leads to a singular or nearly singular variance–covariance matrix that cannot be inverted.
>
> Sustenta o risco de uma covariância singular ou quase singular, que não pode ser invertida.

> [!PDF|yellow] [[todeschini2013_locally-centred-mahalanobis-distance.pdf#page=9&selection=126,0,212,6&color=yellow|todeschini2013_locally-centred-mahalanobis-distance, p.9]]
> > The inverse exists if and only if the determinant of S(j) is larger than zero. This implies that the number of samples n must be greater than the number of variables p and that more than p samples must be different from each other.
>
> Sustenta as condições para a covariância ser invertível. O MCD com o suporte padrão violou a segunda delas na base de cartões, porque o núcleo escolhido ficou concentrado num subespaço.

> [!PDF|yellow] [[evans2001_statistical-aspects-detection-fraud.pdf#page=201&selection=46,36,49,7&color=yellow|evans2001_statistical-aspects-detection-fraud, p.201]]
> > The use of Mahalanobis distances is not justified on a routine basis, but they are very simple to obtain with modern statistical software, and plotting them by treatment group is not arduous
>
> Página 202 do livro. Sustenta a facilidade de obter a Mahalanobis com os programas estatísticos atuais.

> [!quote] Sem PDF: `laca_runtime-optimization-mahalanobis-credit-card-fraud.txt`, resumo
> > Preprocessing steps included feature selection and covariance regularization to ensure numerical stability. [...] Results show that LU Decomposition reduces runtime by 41.82% compared to Normal Inversion, attributed to its ability to factorize matrices and solve systems efficiently, avoiding computationally expensive matrix inversion.
>
> Sustenta, na mesma base de cartões, a regularização da covariância para estabilidade numérica e a vantagem de fatorar a matriz em vez de invertê-la. Sem o artigo completo, serve só como indicação.

## A distância no HDBSCAN\*

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=9&selection=65,2,96,5&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.9]]
> > In addition, let D be (conceptually only) an n × n symmetric matrix containing the distances d(xp, xq) between pairs of objects of X in a metric space
>
> Página 5:8 da revista. Sustenta que o HDBSCAN\* é definido sobre as distâncias entre pares de objetos num espaço métrico qualquer, sem exigir a euclidiana.

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=35&selection=77,57,78,50&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.35]]
> > We used the Cosine measure as dissimilarity function for these datasets.
>
> Página 5:34 da revista. Sustenta que os próprios autores aplicaram o HDBSCAN\* com outra medida de dissimilaridade.

## Ressalvas sobre Xie e Huang (2024)

> [!PDF|yellow] [[xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud.pdf#page=4&selection=26,7,28,26&color=yellow|xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud, p.4]]
> > The Mahalanobis distance is more robust to anomalous values, and it can reduce the impact of anomalies on the distance calculation [14].
>
> Página 162791 da revista. A afirmação contraria Campello et al. (2015), Ghorbani (2019) e De Maesschalck et al. (2000), para quem as estimativas clássicas da média e da covariância são sensíveis às anomalias. A referência [14] (Li, 2019) trata justamente de estimação robusta. Não usar como fonte de robustez.

> [!PDF|yellow] [[xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud.pdf#page=6&selection=37,32,39,26&color=yellow|xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud, p.6]]
> > The original data set was sampled, and then the data set was subjected to data division after the sampled data set
>
> Página 162793 da revista. A reamostragem veio antes da divisão em treino e teste: as amostras sintéticas, interpoladas entre fraudes, podem cair dos dois lados da divisão.

> [!PDF|yellow] [[xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud.pdf#page=8&selection=243,21,244,62&color=yellow|xie2024_mahalanobis-smote-enn-random-forest-credit-card-fraud, p.8]]
> > 5 of the 10 pieces of training of the model achieved excellent results of 1 for all model indicators
>
> Página 162795 da revista. Métricas iguais a 1 numa base com 0,17% de fraudes são compatíveis com o vazamento do trecho anterior. Os resultados não servem de comparação.
