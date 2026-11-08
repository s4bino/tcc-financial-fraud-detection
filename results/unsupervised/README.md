# Resultados do HDBSCAN*

Execuções do `src/models/unsupervised/run_hdbscan_enhanced_pipeline.py` no Colab, copiadas de
`G:\Meu Drive\RESULTS`. Cada pasta reúne uma combinação de base e distância, com os arquivos que
o pipeline grava:

| Arquivo | Conteúdo |
|---|---|
| `hdbscan_enhanced_grid_results_by_fold.csv` | métricas de cada combinação da grade em cada fold interno |
| `hdbscan_enhanced_grid_best_summary.csv` | combinação escolhida pelo DISCO em cada fold interno |
| `best_params_hdbscan_enhanced.json` | os mesmos parâmetros, no formato lido pelo ciclo externo |
| `hdbscan_enhanced_outer_final_report.csv` | métricas no teste de cada fold externo e a média |
| `run.log` | saída da execução |

## `hdbscan_full_mahalanobis_mcd/`

Base completa, distância de Mahalanobis com a covariância do MCD (`support_fraction=1`, amostra de
até 30 mil pontos, piso dos autovalores 1e-9). Grade `min_cluster_size` 50 e `min_samples` 50,
100, 200 e 300, seleção pelo DISCO, `hour_sin` e `hour_cos` fora da distância, `log_amount`
padronizado. Os alertas usam o grau de anomalia com o GLOSH e a taxa 0,0008, e o relatório
acompanha também as taxas 0,0005, 0,0012, 0,0017, 0,0025 e 0,0035 e a AUPRC do GLOSH, do grau de
anomalia com a pertinência e da distância ao centro. Executada em 06/11/2026. No Drive, as pastas
`grid_results_hdbscan_degree_glosh_disco_no_hour_mcd` e
`final_reports_hdbscan_degree_glosh_disco_no_hour_mcd`.
