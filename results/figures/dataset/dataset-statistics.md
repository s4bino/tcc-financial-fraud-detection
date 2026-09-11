# Sumário estatístico da base

Gerado por `src/analysis/dataset_analysis.py`.

## Composição

- Transações: 284.807
- Fraudes: 492 (0.1727%)
- Desbalanceamento: 1 fraude para 577 legítimas
- Período coberto: 48.0 horas

## Ortogonalidade das componentes do PCA

- Correlação absoluta máxima entre V1–V28: 2.32e-14
- Correlação absoluta média: 6.05e-16

Confirma que V1–V28 são mutuamente ortogonais, como se espera de
componentes principais. Nenhum tratamento de multicolinearidade é
necessário entre elas.

## Métrica adotada

O poder discriminante é medido por **AUPRC** (precisão média), não por
AUC ROC. Com 0.173% de fraudes, o AUC ROC é dominado
pela ordenação dos negativos e superestima a separação; o AUPRC mede a
precisão no topo do ranking, que é o que interessa quando apenas uma
fração mínima das transações será acusada.

Um ordenador aleatório obtém AUPRC de 0.001727. O **lift** é o
AUPRC dividido por esse valor: lift 1 significa nenhuma informação.

## A variável Time

- Correlação com Class: -0.0123
- AUPRC isolada: 0.0025 (lift de 1.43×)
- Posição no ranking de poder discriminante: 27 de 30
- Kolmogorov–Smirnov sobre a hora do dia: D = 0.1969, p = 3.667e-17

## A variável Amount

- Correlação com Class: +0.0056
- AUPRC isolada: 0.0042 (lift de 2.45×)
- Mediana: 22.00 (legítimas) contra 9.25 (fraudes)
- Média: 88.29 contra 122.21
- Máximo: 25691.16 contra 2125.87

## Variáveis mais discriminantes

| # | Variável | AUPRC | Lift | Correlação com Class | Sentido |
|---|---|---|---|---|---|
| 1 | V14 | 0.6172 | 357.3× | -0.3025 | menor em fraude |
| 2 | V17 | 0.6155 | 356.3× | -0.3265 | menor em fraude |
| 3 | V12 | 0.5797 | 335.6× | -0.2606 | menor em fraude |
| 4 | V10 | 0.5603 | 324.3× | -0.2169 | menor em fraude |
| 5 | V11 | 0.4946 | 286.3× | +0.1549 | maior em fraude |
| 6 | V16 | 0.4752 | 275.1× | -0.1965 | menor em fraude |
| 7 | V18 | 0.3398 | 196.7× | -0.1115 | menor em fraude |
| 8 | V9 | 0.3347 | 193.7× | -0.0977 | menor em fraude |
| 9 | V3 | 0.2282 | 132.1× | -0.1930 | menor em fraude |
| 10 | V4 | 0.2053 | 118.8× | +0.1334 | maior em fraude |

## Densidade local

- Eixos usados nas projeções: V14, V17, V12
- Mediana da distância ao 20º vizinho: 2.585 (legítimas) contra 7.921 (fraudes)
- Razão: 3.06×
- Kolmogorov–Smirnov: D = 0.7926, p = 4.941e-323

Uma razão acima de 1 indica que as fraudes estão, em mediana, mais
distantes de seus vizinhos — ou seja, em regiões menos densas. É a
premissa que sustenta o tratamento de fraude como ruído.

## Correlação com a variável alvo

Cinco maiores em valor absoluto:

- V17: -0.3265
- V14: -0.3025
- V12: -0.2606
- V10: -0.2169
- V16: -0.1965
