# Figuras e estatísticas

Geradas pelos scripts em `src/analysis/`. Cada figura sai em PNG (leitura
rápida) e PDF (vetorial, para inclusão no LaTeX). Os sumários em Markdown
trazem os números que as figuras mostram.

```bash
python src/analysis/dataset_analysis.py   # results/figures/dataset
python src/analysis/fold_analysis.py      # results/figures/folds
python src/analysis/hdbscan_metrics.py    # results/figures/hdbscan
```

---

## `dataset/` — comportamento da base

| Figura | O que responde |
|---|---|
| `01-correlation-matrix` | estrutura de correlação entre as 31 variáveis |
| `02-feature-discriminative-power` | quanto cada variável separa fraude, sozinha |
| `03-time-relevance` | a variável Time é relevante? |
| `04-class-distributions` | como as seis variáveis mais informativas se distribuem |
| `05-projections` | onde as fraudes ficam, em duas dimensões |
| `06-projection-3d` | a mesma nuvem em três dimensões, de quatro ângulos |
| `07-local-density` | fraudes ocupam regiões menos densas? |

Números em `dataset-statistics.md`; ranking completo em
`feature-discriminative-power.csv`.

**Métrica adotada.** O poder discriminante é medido por **AUPRC**, nunca por
AUC ROC. Com 0,172% de fraudes, o AUC ROC é dominado pela ordenação dos
negativos e superestima a separação; o AUPRC mede a precisão no topo do
ranking, que é o que interessa quando só uma fração mínima das transações será
acusada. Um ordenador aleatório obtém AUPRC de 0,001727, e o **lift** é o AUPRC
dividido por esse valor — lift 1 significa nenhuma informação.

**Principais achados.** As componentes V1–V28 são ortogonais entre si
(correlação absoluta máxima de 2,32 × 10⁻¹⁴), o que dispensa qualquer
tratamento de multicolinearidade. As mais discriminantes são V14
(AUPRC 0,617, lift 357×), V17 (0,616, 356×) e V12 (0,580, 336×).

A troca de métrica reordena o ranking de forma relevante: **V17 sobe da oitava
para a segunda posição** — é a variável de maior correlação absoluta com a
classe, e o AUC ROC a escondia — enquanto **V4 cai da segunda para a décima**
(AUPRC 0,205). V4 ordena bem os negativos, mas não coloca os positivos no topo.

A premissa do trabalho se confirma: a distância mediana ao 20º vizinho é
**3,06 vezes maior** para fraudes do que para transações legítimas
(7,92 contra 2,59; Kolmogorov–Smirnov D = 0,793). Fraudes estão, de fato, em
regiões de baixa densidade.

**Sobre a variável Time**, a resposta tem duas partes. Como ordenador isolado
ela é quase inútil: AUPRC de 0,0025, lift de apenas **1,43×**, 27ª posição
entre 30 variáveis, correlação de −0,012 com a classe. Mas a hora do dia
derivada dela carrega risco real — a faixa das 03h às 06h concentra 58,7
fraudes por 10.000 transações contra 10,4 na faixa das 21h às 24h, uma razão de
**5,6×**. O lift baixo não contradiz isso: a maioria das fraudes ocorre de dia,
junto com a maioria das transações, de modo que a hora serve para multiplicar a
probabilidade a priori, não para ordenar transações uma a uma.

---

## `folds/` — as partições da validação cruzada

| Figura | O que responde |
|---|---|
| `01-fold-composition` | a estratificação se manteve? |
| `02-fold-coverage` | cada transação é testada exatamente uma vez? |
| `03-fraud-density-by-fold` | a premissa de baixa densidade vale em todos os folds? |

Números em `fold-statistics.md`.

**Principais achados.** A estratificação se manteve: a proporção de fraude
varia de 0,1720% a 0,1738% entre partições, amplitude de 0,0018 ponto
percentual. Todas as 284.807 transações são testadas exatamente uma vez, e a
sobreposição entre conjuntos de teste é nula.

A razão de densidade entre fraude e transação legítima é **notavelmente
estável** entre os folds: de 2,93× a 3,03×. Isso importa para a interpretação
dos resultados do HDBSCAN, tratada a seguir.

---

## `hdbscan/` — variação das métricas

Substitui a figura `auc-variation` anterior, que mostrava apenas AUC ROC e
tinha títulos sobrepostos. O AUC ROC foi descartado das figuras pelo motivo
registrado acima; as métricas acompanhadas são DBCV, AUPRC, precisão, F1 e
revocação.

| Figura | O que responde |
|---|---|
| `01-metric-stability` | quanto cada métrica varia entre folds, por configuração |
| `02-dbcv-spread-by-config` | quais configurações são estáveis e quais não |
| `03-dbcv-vs-performance` | o DBCV prediz o desempenho supervisionado? |

Números em `hdbscan-metric-variation.md`.

**Principais achados.** A instabilidade é **concentrada, não difusa**. A
amplitude mediana do DBCV entre folds é de apenas 0,065; mas as cinco
configurações com `min_cluster_size=5` chegam a 0,852. Todas as demais são
estáveis. O problema não é o HDBSCAN em geral — é essa região específica do
espaço de parâmetros, que por infelicidade é a que vence a seleção.

Cruzando com a análise por fold: como a estrutura de densidade das partições é
praticamente idêntica (2,93× a 3,03×), **a instabilidade vem do algoritmo, não
dos dados**.

Sobre o critério de seleção, o DBCV correlaciona-se positivamente com as
métricas supervisionadas — ρ de Spearman de +0,70 no conjunto completo e +0,65
dentro do regime de DBCV baixo, todos com p abaixo de 10⁻¹⁵. A ressalva está
registrada no sumário: a distribuição do DBCV é bimodal, e o coeficiente
global mede em boa parte a distância entre os dois regimes. O coeficiente
interno ao regime inferior é o que sustenta a afirmação de que há ordenação, e
não apenas separação.

Por fim, máximo e média entre folds elegem a **mesma** configuração
(`min_cluster_size=5`, `min_samples=300`), o que torna a escolha do critério
indiferente para esta grade.
