# Variação das métricas do HDBSCAN

Gerado por `src/analysis/hdbscan_metrics.py`.

- Combinações avaliadas: 25
- Folds: [1, 2, 3, 4, 5]
- Execuções registradas: 125

## Amplitude por métrica

Medida dentro de cada configuração, entre os cinco folds, e depois
resumida pelo maior e pelo menor caso.

| Métrica | Menor amplitude | Maior amplitude | Amplitude mediana |
|---|---|---|---|
| DBCV | 0.0130 | 0.8524 | 0.0646 |
| AUPRC | 0.0000 | 0.0318 | 0.0031 |
| Precisão | 0.0000 | 0.0339 | 0.0031 |
| F1 | 0.0001 | 0.0650 | 0.0061 |
| Revocação | 0.0127 | 0.1013 | 0.0380 |

## Configurações extremas em DBCV

- Mais estável: `min_cluster_size=50`, `min_samples=300` — DBCV de 0.0992 a 0.1122 (amplitude 0.0130)
- Menos estável: `min_cluster_size=5`, `min_samples=300` — DBCV de 0.1066 a 0.9590 (amplitude 0.8524)

## O DBCV prediz o desempenho?

| Métrica | ρ global | p | ρ no regime inferior | p |
|---|---|---|---|---|
| auprc | +0.7044 | 4.975e-20 | +0.6574 | 4.664e-16 |
| precision | +0.6978 | 1.539e-19 | +0.6497 | 1.311e-15 |
| f1 | +0.6983 | 1.399e-19 | +0.6504 | 1.201e-15 |

**Atenção à interpretação.** A distribuição do DBCV é bimodal: um
grupo denso abaixo de 0,2 e um grupo destacado acima de 0,95, sem
nenhuma observação entre os dois. O coeficiente global mede
sobretudo a distância entre esses dois regimes, não uma ordenação
gradual. A leitura defensável é que o DBCV **distingue corretamente
os dois regimes** — e o regime que ele premia é o de melhor
desempenho supervisionado. Afirmar que o DBCV ordena configurações
por qualidade exige olhar o coeficiente calculado dentro do regime
inferior, apresentado na última coluna.

## Seleção: máximo contra média

- Vencedora pelo **máximo** entre folds: `min_cluster_size=5`, `min_samples=300`
- Vencedora pelo **DBCV médio** entre folds: `min_cluster_size=5`, `min_samples=300`

As duas coincidem.
