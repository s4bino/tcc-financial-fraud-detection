> [!check] Status: em uso na monografia
> - `joanes1998`: consta do `refbib.bib`, citado em Metodologia › Pré-processamento como fonte da definição do coeficiente de assimetria amostral $g_1$.
> - `abdi2010` e `huang2015`, também citados na seção, têm os trechos nas notas de PCA e de escalonamento.
> - `dalpozzolo2015`: consta do `refbib.bib`, citado em Metodologia › Nested Cross-Validation para a subamostragem como técnica para bases desbalanceadas, a premissa das observações redundantes da classe majoritária e o viés que não afeta a ordenação, mas afeta a classificação e a calibração. Os grifos do limiar ficam na nota de detecção de fraudes.

> [!PDF|yellow] [[joanes1998_comparing-measures-sample-skewness-kurtosis.pdf#page=1&selection=39,0,49,12&color=yellow|joanes1998_comparing-measures-sample-skewness-kurtosis, p.1]]
> > We consider the traditional measures of skewness and kurtosis, g1 and g2 respectively
>
> Página 183 da revista. Sustenta que $g_1$ é a medida tradicional de assimetria amostral.

^joanes1998-p1

> [!PDF|yellow] [[joanes1998_comparing-measures-sample-skewness-kurtosis.pdf#page=2&selection=0,7,60,1&color=yellow|joanes1998_comparing-measures-sample-skewness-kurtosis, p.2]]
> > is typical of many older texts in defining such coefficients as $g_1 = m_3/m_2^{3/2}$ and $g_2 = m_4/m_2^2 - 3$, where the sample moments for samples of size n are given by $m_r = \frac{1}{n}\sum(x_i - \bar{x})^r$.
>
> Página 184 da revista. Sustenta a equação do coeficiente de assimetria: o momento amostral de ordem 3 dividido pelo de ordem 2 elevado a 3/2. É a definição calculada pelo `scipy.stats.skew` na configuração padrão, usada para obter os valores de Amount.

^joanes1998-p2

## Base subamostrada

> [!PDF|yellow] [[dalpozzolo2015_calibrating-probability-undersampling.pdf#page=1&selection=36,1,37,57&color=yellow|dalpozzolo2015_calibrating-probability-undersampling, p.1]]
> > Undersampling is a popular technique for unbalanced datasets to reduce the skew in class distributions.
>
> Resumo. Sustenta a subamostragem como técnica comum em bases desbalanceadas para reduzir a diferença entre as proporções das classes.

^dalpozzolo2015-p1

> [!PDF|yellow] [[dalpozzolo2015_calibrating-probability-undersampling.pdf#page=1&selection=100,46,104,13&color=yellow|dalpozzolo2015_calibrating-probability-undersampling, p.1]]
> > The assumption behind this strategy is that in the majority class there are many redundant observations and randomly removing some of them does not change the estimation of the within-class distribution.
>
> Seção I. Sustenta a premissa da subamostragem aleatória, conferida pelo teste de Kolmogorov-Smirnov entre as legítimas mantidas e as removidas.

^dalpozzolo2015-p1-2

> [!PDF|yellow] [[dalpozzolo2015_calibrating-probability-undersampling.pdf#page=1&selection=75,15,78,12&color=yellow|dalpozzolo2015_calibrating-probability-undersampling, p.1]]
> > Although the bias due to undersampling does not affect the ranking order returned by the posterior probability, it significantly impacts the classification accuracy and probability calibration.
>
> Resumo. Sustenta que o viés da subamostragem não afeta a ordenação das transações, mas afeta a classificação e a calibração das probabilidades, o que motiva os pesos das legítimas.

^dalpozzolo2015-p1-3

