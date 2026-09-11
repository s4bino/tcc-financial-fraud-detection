# Sumário das partições

Gerado por `src/analysis/fold_analysis.py`.

## Estratificação

| Fold | Partição | Amostras | Fraudes | Proporção (%) |
|---|---|---|---|---|
| 1 | Teste externo | 56.962 | 99 | 0.1738 |
| 1 | Treino externo | 227.845 | 393 | 0.1725 |
| 1 | Treino interno | 182.276 | 314 | 0.1723 |
| 1 | Validação interna | 45.569 | 79 | 0.1734 |
| 2 | Teste externo | 56.962 | 99 | 0.1738 |
| 2 | Treino externo | 227.845 | 393 | 0.1725 |
| 2 | Treino interno | 182.276 | 314 | 0.1723 |
| 2 | Validação interna | 45.569 | 79 | 0.1734 |
| 3 | Teste externo | 56.961 | 98 | 0.1720 |
| 3 | Treino externo | 227.846 | 394 | 0.1729 |
| 3 | Treino interno | 182.276 | 315 | 0.1728 |
| 3 | Validação interna | 45.570 | 79 | 0.1734 |
| 4 | Teste externo | 56.961 | 98 | 0.1720 |
| 4 | Treino externo | 227.846 | 394 | 0.1729 |
| 4 | Treino interno | 182.276 | 315 | 0.1728 |
| 4 | Validação interna | 45.570 | 79 | 0.1734 |
| 5 | Teste externo | 56.961 | 98 | 0.1720 |
| 5 | Treino externo | 227.846 | 394 | 0.1729 |
| 5 | Treino interno | 182.276 | 315 | 0.1728 |
| 5 | Validação interna | 45.570 | 79 | 0.1734 |

Amplitude da proporção de fraude dentro de cada tipo de partição:

- Teste externo: 0.1720% a 0.1738% (amplitude de 0.0018 ponto percentual)
- Treino externo: 0.1725% a 0.1729% (amplitude de 0.0004 ponto percentual)
- Treino interno: 0.1723% a 0.1728% (amplitude de 0.0005 ponto percentual)
- Validação interna: 0.1734% a 0.1734% (amplitude de 0.0000 ponto percentual)

## Cobertura e isolamento

- Transações testadas 1 vez(es): 284.807
- Sobreposição máxima fora da diagonal: 0.00%

Sobreposição nula fora da diagonal confirma que os conjuntos de teste
são mutuamente exclusivos. Toda transação testada exatamente uma vez
confirma a cobertura integral da base.

## Densidade local das fraudes por fold

| Fold | Fraudes | Mediana legítima | Mediana fraude | Razão |
|---|---|---|---|---|
| 1 | 393 | 2.798 | 8.198 | 2.93× |
| 2 | 393 | 2.795 | 8.187 | 2.93× |
| 3 | 394 | 2.798 | 8.334 | 2.98× |
| 4 | 394 | 2.756 | 8.323 | 3.02× |
| 5 | 394 | 2.809 | 8.504 | 3.03× |

Amplitude da razão entre folds: 2.93× a 3.03×.

Se a razão for estável entre os folds, a instabilidade observada nos
resultados do HDBSCAN não vem de diferenças na estrutura de densidade
das partições.
