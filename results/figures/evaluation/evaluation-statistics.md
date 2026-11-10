# Resultados: capítulo 5 e apêndice A

Gerado por `src/analysis/results_analysis.py` a partir das rodadas finais.

## Teste externo: média e desvio padrão nos cinco folds

| Base | Modelo | AUPRC | Precisão | Revocação | F1 |
|---|---|---|---|---|---|
| full | rf | 0.852 ± 0.024 | 0.945 ± 0.022 | 0.803 ± 0.017 | 0.868 ± 0.016 |
| undersampled | rf | 0.841 ± 0.024 | 0.869 ± 0.025 | 0.821 ± 0.015 | 0.844 ± 0.018 |
| full | xgb | 0.852 ± 0.029 | 0.944 ± 0.045 | 0.783 ± 0.040 | 0.856 ± 0.039 |
| undersampled | xgb | 0.853 ± 0.021 | 0.918 ± 0.023 | 0.811 ± 0.029 | 0.861 ± 0.020 |
| full | hdbscan_euclidean | 0.238 ± 0.025 | 0.315 ± 0.018 | 0.317 ± 0.059 | 0.314 ± 0.029 |
| undersampled | hdbscan_euclidean | 0.183 ± 0.020 | 0.245 ± 0.033 | 0.230 ± 0.019 | 0.235 ± 0.011 |
| full | hdbscan_mahalanobis | 0.210 ± 0.019 | 0.254 ± 0.011 | 0.256 ± 0.042 | 0.254 ± 0.022 |
| undersampled | hdbscan_mahalanobis | 0.086 ± 0.008 | 0.127 ± 0.023 | 0.116 ± 0.024 | 0.120 ± 0.021 |
| full | hdbscan_mahalanobis_mcd | 0.708 ± 0.031 | 0.792 ± 0.056 | 0.801 ± 0.011 | 0.796 ± 0.034 |
| undersampled | hdbscan_mahalanobis_mcd | 0.700 ± 0.031 | 0.794 ± 0.045 | 0.801 ± 0.011 | 0.797 ± 0.027 |

## Alertas somados nos cinco testes externos

| Base | Modelo | Alertas | VP | FP | FN |
|---|---|---|---|---|---|
| full | hdbscan_euclidean | 496 | 156 | 340 | 336 |
| full | hdbscan_mahalanobis | 496 | 126 | 370 | 366 |
| full | hdbscan_mahalanobis_mcd | 499 | 394 | 105 | 98 |
| full | rf | 418 | 395 | 23 | 97 |
| full | xgb | 408 | 385 | 23 | 107 |
| undersampled | hdbscan_euclidean | 471 | 113 | 358 | 379 |
| undersampled | hdbscan_mahalanobis | 453 | 57 | 396 | 435 |
| undersampled | hdbscan_mahalanobis_mcd | 497 | 394 | 103 | 98 |
| undersampled | rf | 465 | 404 | 61 | 88 |
| undersampled | xgb | 435 | 399 | 36 | 93 |

## Fração de alertas do HDBSCAN* com mahalanobis_mcd

Fração adotada nas tabelas: 0.0017. Fração dos alertas das rodadas: 0.0008.

| Base | Fração | Precisão | Revocação | F1 |
|---|---|---|---|---|
| full | 0.0005 | 0.822 | 0.252 | 0.384 |
| full | 0.0008 | 0.812 | 0.384 | 0.520 |
| full | 0.0012 | 0.847 | 0.596 | 0.697 |
| full | 0.0017 | 0.792 | 0.801 | 0.796 |
| full | 0.0025 | 0.540 | 0.801 | 0.645 |
| full | 0.0035 | 0.394 | 0.803 | 0.528 |
| undersampled | 0.0005 | 0.827 | 0.260 | 0.395 |
| undersampled | 0.0008 | 0.778 | 0.364 | 0.496 |
| undersampled | 0.0012 | 0.834 | 0.583 | 0.685 |
| undersampled | 0.0017 | 0.794 | 0.801 | 0.797 |
| undersampled | 0.0025 | 0.567 | 0.801 | 0.663 |
| undersampled | 0.0035 | 0.397 | 0.803 | 0.531 |

## Escores de referência do HDBSCAN* (AUPRC)

| Base | Distância | Grau de anomalia | GLOSH | Distância ao centro |
|---|---|---|---|---|
| full | euclidean | 0.238 ± 0.025 | 0.239 ± 0.026 | 0.166 ± 0.020 |
| full | mahalanobis | 0.210 ± 0.019 | 0.210 ± 0.019 | 0.169 ± 0.019 |
| full | mahalanobis_mcd | 0.708 ± 0.031 | 0.709 ± 0.030 | 0.713 ± 0.028 |
| undersampled | euclidean | 0.183 ± 0.020 | 0.184 ± 0.020 | 0.165 ± 0.020 |
| undersampled | mahalanobis | 0.086 ± 0.008 | 0.087 ± 0.008 | 0.082 ± 0.008 |
| undersampled | mahalanobis_mcd | 0.700 ± 0.031 | 0.703 ± 0.031 | 0.712 ± 0.028 |

## Ciclo interno dos modelos supervisionados

| Modelo | Base | Configuração | Fold que a indicou | AUPRC no fold | AUPRC média | Maior média da grade | Combinações a menos de 0,01 dela | Combinações distintas indicadas |
|---|---|---|---|---|---|---|---|---|
| rf | full | n_estimators=300, max_features=0.5, class_weight=None, max_samples=None, min_samples_leaf=1 | 1 | 0.920 | 0.885 | 0.887 | 39 de 162 | 5 de 5 |
| rf | undersampled | n_estimators=300, max_features=0.5, class_weight=None, max_samples=None, min_samples_leaf=2 | 1 | 0.916 | 0.867 | 0.872 | 21 de 162 | 5 de 5 |
| xgb | full | n_estimators=300, max_depth=6, learning_rate=0.1, gamma=0 | 1 | 0.914 | 0.886 | 0.886 | 12 de 81 | 3 de 5 |
| xgb | undersampled | n_estimators=100, max_depth=9, learning_rate=0.1, gamma=0 | 1 | 0.919 | 0.869 | 0.878 | 42 de 81 | 4 de 5 |

## Ciclo interno do HDBSCAN*

| Distância | Base | m_pts | Grupos | DISCO médio | DISCO máximo | AUPRC | Escolhida |
|---|---|---|---|---|---|---|---|
| euclidean | full | 50 | 124.4 | -0.162 | -0.142 | 0.243 |  |
| euclidean | full | 100 | 61.4 | -0.087 | -0.078 | 0.281 |  |
| euclidean | full | 200 | 33.0 | -0.161 | -0.133 | 0.321 |  |
| euclidean | full | 300 | 3.8 | 0.194 | 0.212 | 0.336 | sim |
| euclidean | undersampled | 50 | 39.6 | -0.073 | -0.030 | 0.194 |  |
| euclidean | undersampled | 100 | 25.2 | -0.130 | -0.103 | 0.237 |  |
| euclidean | undersampled | 200 | 4.8 | 0.111 | 0.207 | 0.281 | sim |
| euclidean | undersampled | 300 | 3.0 | 0.177 | 0.190 | 0.296 |  |
| mahalanobis | full | 50 | 122.6 | -0.122 | -0.064 | 0.200 |  |
| mahalanobis | full | 100 | 58.4 | -0.104 | -0.096 | 0.232 |  |
| mahalanobis | full | 200 | 38.8 | -0.172 | -0.167 | 0.288 |  |
| mahalanobis | full | 300 | 6.8 | 0.100 | 0.110 | 0.320 | sim |
| mahalanobis | undersampled | 50 | 40.2 | -0.087 | -0.073 | 0.127 |  |
| mahalanobis | undersampled | 100 | 5.4 | 0.179 | 0.238 | 0.134 | sim |
| mahalanobis | undersampled | 200 | 6.0 | 0.074 | 0.078 | 0.145 |  |
| mahalanobis | undersampled | 300 | 5.4 | 0.064 | 0.179 | 0.155 |  |
| mahalanobis_mcd | full | 50 | 60.0 | 0.198 | 0.731 | 0.033 |  |
| mahalanobis_mcd | full | 100 | 5.0 | 0.694 | 0.727 | 0.467 |  |
| mahalanobis_mcd | full | 200 | 3.0 | 0.708 | 0.736 | 0.763 | sim |
| mahalanobis_mcd | full | 300 | 3.0 | 0.707 | 0.736 | 0.773 |  |
| mahalanobis_mcd | undersampled | 50 | 5.0 | 0.637 | 0.661 | 0.056 |  |
| mahalanobis_mcd | undersampled | 100 | 4.0 | 0.630 | 0.656 | 0.500 |  |
| mahalanobis_mcd | undersampled | 200 | 2.0 | 0.655 | 0.675 | 0.778 | sim |
| mahalanobis_mcd | undersampled | 300 | 2.0 | 0.653 | 0.674 | 0.775 |  |
