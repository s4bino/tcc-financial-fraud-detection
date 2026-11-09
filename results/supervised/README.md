# Resultados dos modelos supervisionados

## Rodada final

Execuções do `src/models/supervised/run_supervised_grid_search.py` no Colab, copiadas de
`G:\Meu Drive\RESULTS\supervised` sem mudar de nome. Cada pasta reúne um modelo e uma base,
`rf` ou `xgb`, com o sufixo `_undersampled` na base subamostrada. As quatro deixam `hour_sin` e
`hour_cos` fora do modelo, mantêm as transações duplicadas e escolhem a configuração pela AUPRC.
O `run_config.json` de cada pasta registra a grade e as pastas de folds usadas.

O ciclo interno avalia toda a grade nos 5 folds internos, com 162 combinações no Random Forest e
81 no XGBoost. Cada fold interno indica a sua melhor combinação, e a de maior AUPRC entre elas é
aplicada aos 5 folds externos, como no HDBSCAN*.

| Arquivo | Conteúdo |
|---|---|
| `results_by_fold.csv` | métricas de cada combinação da grade em cada fold interno |
| `results_summary.csv` | média das métricas de cada combinação nos folds internos |
| `<modelo>_outer_final_report.csv` | métricas no teste de cada fold externo, a média e o desvio padrão |
| `run_config.json` | configuração da execução |
| `run.log` | saída da execução, com a melhor combinação de cada fold interno |
