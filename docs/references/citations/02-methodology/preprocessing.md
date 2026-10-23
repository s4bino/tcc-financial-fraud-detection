> [!check] Status: em uso na monografia
> - `joanes1998`: consta do `refbib.bib`, citado em Metodologia › Pré-processamento como fonte da definição do coeficiente de assimetria amostral $g_1$.
> - `abdi2010` e `huang2015`, também citados na seção, têm os trechos nas notas de PCA e de escalonamento.

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

