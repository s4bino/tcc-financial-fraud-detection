# Resultados do HDBSCAN*

## Rodada final

Execuções do `src/models/unsupervised/run_hdbscan_enhanced_pipeline.py` no Colab, copiadas de
`G:\Meu Drive\RESULTS\unsupervised` sem mudar de nome. Cada pasta reúne uma combinação de base e
distância, `hdbscan_<distância>`, com o sufixo `_undersampled` na base subamostrada. Todas usam a
configuração fixa em `build_config`, e o `run_config.json` de cada pasta registra a execução.

| Arquivo | Conteúdo |
|---|---|
| `hdbscan_enhanced_grid_results_by_fold.csv` | métricas de cada combinação da grade em cada fold interno |
| `hdbscan_enhanced_grid_best_summary.csv` | combinação escolhida pelo DISCO em cada fold interno |
| `best_params_hdbscan_enhanced.json` | os mesmos parâmetros, no formato lido pelo ciclo externo |
| `hdbscan_enhanced_outer_final_report.csv` | métricas no teste de cada fold externo, a média e o desvio padrão |
| `run_config.json` | configuração da execução |
| `run.log` | saída da execução |

### `hdbscan_mahalanobis_mcd/`

Base completa, distância de Mahalanobis com a covariância do MCD. Executada em 06/11/2026, antes
de `build_config` existir, com a mesma configuração dele: grade `min_cluster_size` 50 e
`min_samples` 50, 100, 200 e 300, seleção pelo DISCO, `hour_sin` e `hour_cos` fora da distância,
alertas pelo grau de anomalia com o GLOSH na taxa 0,0008 e as taxas 0,0005 a 0,0035 só
registradas. Por isso não tem `run_config.json`, e o relatório externo termina só com a média.
No Drive, as pastas `grid_results_hdbscan_degree_glosh_disco_no_hour_mcd` e
`final_reports_hdbscan_degree_glosh_disco_no_hour_mcd`.

## Outros arquivos

- `hdbscan_grid_results_by_fold.csv` e `hdbscan_grid_best_summary.csv`: grade do HDBSCAN*
  original, de setembro de 2026, lida por `src/analysis/hdbscan_metrics.py`.
- `disco/`: medições do DISCO, gravadas por `src/analysis/disco_benchmark.py`.
