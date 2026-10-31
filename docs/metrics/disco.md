# DISCO

O DISCO (*Density-based Internal Score for Clusterings with nOise*) é um índice interno de validação de agrupamentos por densidade, proposto por Beer et al. (2025). Ele mede a qualidade de uma partição sem rótulos verdadeiros e, ao contrário dos demais índices, avalia também a qualidade dos rótulos de ruído.

| | |
|---|---|
| **Função** | `disco_score(X, labels, min_points=5, tree=None)` |
| **Módulo** | [`src/models/disco.py`](../../src/models/disco.py) |
| **Faixa de valores** | de -1 a 1 |
| **Melhor valor** | 1 (quanto maior, melhor) |
| **Usa rótulos verdadeiros** | não |
| **Determinístico** | sim |
| **Parâmetro** | `min_points` (μ no artigo), 5 por padrão |
| **Dependências** | NumPy, SciPy e scikit-learn, sem GPU |

## Uso

```python
from disco import disco_score

score = disco_score(X, labels)
```

`X` tem forma `(n_pontos, n_variáveis)`, e `labels` traz o rótulo de cada ponto, com `-1` para o ruído. Os identificadores dos agrupamentos podem ser quaisquer outros valores.

Para o escore de cada ponto, use `disco_samples`, útil para examinar agrupamentos ou rótulos de ruído em separado:

```python
from disco import disco_samples

scores = disco_samples(X, labels)
scores[labels == -1].mean()  # qualidade dos rótulos de ruído
```

Para avaliar várias partições do mesmo `X`, construa a árvore uma vez e a reaproveite. A árvore concentra quase todo o custo e não depende dos rótulos:

```python
from disco import DCTree, disco_score

tree = DCTree(X)
scores = [disco_score(X, labels, tree=tree) for labels in partitions]
```

A árvore precisa ter sido construída sobre o mesmo `X`. O número de pontos e o `min_points` são conferidos, e uma divergência levanta `ValueError`. O conteúdo de `X` não é conferido.

### Parâmetros

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `X` | `np.ndarray` | Pontos, comparados pela distância euclidiana. |
| `labels` | `np.ndarray` | Rótulo de cada ponto, com `-1` para o ruído. |
| `min_points` | `int` | Parâmetro μ da distância de núcleo, de 1 ao número de pontos. |
| `tree` | `DCTree` ou `None` | Árvore já construída sobre os mesmos `X` e `min_points`. |

`disco_score` devolve um `float`, e `disco_samples`, um `np.ndarray` na ordem de `X`. Um conjunto vazio, tamanhos diferentes entre `X` e `labels` ou um `min_points` fora da faixa levantam `ValueError`.

## Definição

As equações seguem a numeração de Beer et al. (2025).

### Distância dc

A distância de núcleo $\kappa(x)$ é a distância euclidiana de $x$ ao seu μ-ésimo vizinho mais próximo. A partir dela, a distância de alcançabilidade mútua entre dois pontos é:

$$d_m(x, y) = \max\big(\kappa(x),\ \kappa(y),\ d_{eucl}(x, y)\big)$$

A distância de conectividade por densidade (distância dc) é a maior aresta do caminho entre os dois pontos na árvore geradora mínima do grafo com os pesos $d_m$:

$$d_{dc}(x, y) = \max_{e \in p(x, y)} |e| \quad \text{se } x \neq y, \text{ e } 0 \text{ caso contrário} \tag{1}$$

em que $p(x, y)$ é o caminho entre $x$ e $y$ na árvore. Dois pontos de uma mesma região densa ficam próximos mesmo em um agrupamento alongado, porque o caminho avança por passos curtos.

### Escore da partição

O DISCO da partição é a média dos escores de todos os pontos:

$$\rho(X) = \operatorname*{avg}_{x \in X} \rho(x) \tag{2}$$

O escore de cada ponto depende do seu rótulo:

$$\rho(x) = \begin{cases} \rho_{cluster}(x) & \text{se } x \in C_i \text{ para algum } i \\ \rho_{noise}(x) & \text{se } x \text{ é ruído} \end{cases} \tag{3}$$

### Pontos de agrupamento

O escore é a silhueta calculada sobre a distância dc:

$$\rho_{cluster}(x) = \min_{C_i \neq \hat{C}_x} \frac{\widetilde{d_{dc}}(x, C_i) - \widetilde{d_{dc}}(x, \hat{C}_x)}{\max\big(\widetilde{d_{dc}}(x, C_i),\ \widetilde{d_{dc}}(x, \hat{C}_x)\big)} \tag{4}$$

em que $\hat{C}_x$ é o agrupamento de $x$ e $\widetilde{d_{dc}}(x, C)$ é a distância dc média de $x$ aos pontos de $C$.

### Pontos de ruído

Um rótulo de ruído é bom quando o ponto está em uma região esparsa e longe de todos os agrupamentos. A primeira condição compara a distância de núcleo do ponto com a de cada agrupamento:

$$\rho_{sparse}(x_n) = \min_{C_i} \frac{\kappa(x_n) - \kappa(C_i)}{\max\big(\kappa(x_n),\ \kappa(C_i)\big)} \tag{5}$$

em que $\kappa(C)$ é a maior distância de núcleo entre os pontos de $C$. A segunda condição compara, com esse mesmo limiar, a distância dc até o agrupamento:

$$\rho_{far}(x_n) = \min_{C_i} \frac{\min_{y \in C_i} d_{dc}(x_n, y) - \kappa(C_i)}{\max\big(\min_{y \in C_i} d_{dc}(x_n, y),\ \kappa(C_i)\big)} \tag{6}$$

O escore do ponto é o pior dos dois:

$$\rho_{noise}(x_n) = \min\big(\rho_{sparse}(x_n),\ \rho_{far}(x_n)\big) \tag{7}$$

### Casos especiais

O DISCO está definido para qualquer partição. Sem agrupamentos, todos os pontos valem -1. Com um único agrupamento e sem ruído, e em agrupamentos unitários, o escore é 0. Um denominador nulo, que ocorre com pontos repetidos ao menos μ vezes, também leva a fração a 0. Com um único agrupamento e ruído, a separação é medida até o ruído mais próximo:

$$\rho_{cluster}(x) = \frac{\min_{x_n} d_{dc}(x, x_n) - \widetilde{d_{dc}}(x, \hat{C}_x)}{\max\big(\min_{x_n} d_{dc}(x, x_n),\ \widetilde{d_{dc}}(x, \hat{C}_x)\big)} \tag{8}$$

### Interpretação

| Valor | Ponto de agrupamento | Ponto de ruído |
|---|---|---|
| Próximo de 1 | Compacto e bem separado dos demais agrupamentos. | Em região esparsa e distante dos agrupamentos. |
| Próximo de 0 | Na fronteira entre agrupamentos. | Tão denso quanto o agrupamento mais esparso, ou na borda de um agrupamento. |
| Negativo | Mais próximo de outro agrupamento que do seu. | Em região densa ou dentro de um agrupamento: um rótulo de agrupamento caberia melhor. |

## Escolha de μ

O padrão é μ = 5, valor que Beer et al. (2025, Seção 4.5) fixam em todos os experimentos ao concluir que o índice é pouco sensível a ele. No pipeline, μ não acompanha o `min_samples` do HDBSCAN\*. Assim, todas as combinações da grade são medidas na mesma distância dc, o que torna as notas comparáveis entre si e permite reaproveitar a árvore.

## Implementação

O código oficial dos autores (`src/Evaluation/disco.py` em [pasiweber/DISCO](https://github.com/pasiweber/DISCO)) monta a matriz n × n das distâncias euclidianas e, depois, a das distâncias dc. Esse custo é o que limita o tamanho dos dados: com os 64.316 pontos de um fold subamostrado, cada matriz ocuparia 33 GB. A implementação deste repositório calcula as mesmas quantidades sem nenhuma das duas, em quatro etapas.

1. **Distâncias de núcleo:** busca dos vizinhos com a KD-tree do scikit-learn, exata e paralela.
2. **Árvore geradora mínima:** algoritmo de Prim, calculando as distâncias de cada ponto só quando ele é visitado, o que mantém a memória em O(n).
3. **Árvore de fusões:** as arestas são fundidas em ordem crescente. A distância dc entre dois pontos é a altura da fusão que os une pela primeira vez.
4. **Consultas:** cada ponto está à mesma distância de todos os pontos que entram na sua subárvore por uma fusão, igual à altura dela. Logo, a soma das distâncias de um ponto a um agrupamento se obtém percorrendo os seus ancestrais, com cada altura multiplicada pelo número de pontos do agrupamento no outro ramo. A menor distância vem do ancestral mais baixo cujo outro ramo contém algum ponto do agrupamento. Tabelas de ancestrais fazem esse percurso em O(log h) passos vetorizados, para todos os pontos de uma vez, em que h é a altura da árvore.

### Complexidade

| | Tempo | Memória |
|---|---|---|
| Construção da árvore | O(n² d) | O(n log h) |
| Cada partição, com k agrupamentos | O(k n log h) | O(n log h) |
| Código oficial, por partição | O(n² d) | O(n²) |

O tempo da construção continua quadrático, como no artigo (Seção 3.2). O ganho vem de a memória deixar de ser quadrática e de a parte que depende dos rótulos ficar muito mais barata.

### Diferenças em relação ao código oficial

As definições são as mesmas. As diferenças são estas:

- **Primeiro vizinho:** o artigo define κ pelo μ-ésimo vizinho sem detalhar a contagem. Como no código oficial, o próprio ponto conta como o primeiro.
- **Pontos repetidos:** o código oficial usa `pairwise_distances`, cujo atalho numérico devolveu 8,4e-08 onde a distância exata é 0. Aqui a distância é exata, o que explica a maior diferença na comparação abaixo.
- **`min_points` igual ao número de pontos:** é aceito aqui e levanta erro no código oficial.

O repositório oficial não declara licença. Por isso, o módulo foi escrito a partir das equações do artigo, e o código oficial serviu só como referência para a comparação.

## Uso no pipeline

`run_hdbscan_enhanced_pipeline.py` registra a métrica em `VALIDATION_METRICS` com o nome `"disco"`:

```python
FOLDS = paths.UNDERSAMPLED_PATH

CONFIG = PipelineConfig(
    ...
    disco_min_points=5,
    selection_metric="disco",
    extra_validation_metrics=("dbcv",),
)
```

O ciclo interno escolhe, em cada fold, a combinação de maior `disco_score`, e o DBCV fica registrado na coluna `dbcv_score`, sem papel na seleção. O DISCO usa o treino inteiro, enquanto o DBCV segue limitado a uma amostra de 28.000 pontos. A árvore é construída uma vez por fold e reaproveitada entre as combinações. Os resultados vão para `grid_results_hdbscan_glosh_disco_undersampled` e `final_reports_hdbscan_glosh_disco_undersampled`.

O DBCV do pipeline converte qualquer erro em -1. O DISCO não faz isso: como está definido para qualquer partição, um erro indica um defeito e interrompe a execução. Uma partição só de ruído vale -1 pela própria definição.

Os demais scripts seguem com o DBCV. Para usar a métrica em outro script, basta importar `disco_score`.

## Resultados

Medições em CPU, em uma máquina de 12 threads e 16 GB de memória, sobre o treino do fold interno 1 da base subamostrada (64.316 pontos e 31 variáveis). As tabelas completas estão em [`results/unsupervised/disco/`](../../results/unsupervised/disco/).

### Equivalência com o código oficial

A comparação ponto a ponto com o código oficial cobriu 25 cenários: partições comuns, μ de 1 a 100, um único agrupamento com ruído, só ruído, agrupamentos unitários, empates, pontos repetidos e amostras do fold real com rótulos do HDBSCAN e aleatórios. A maior diferença em um ponto foi de 3,3e-10, no cenário de pontos repetidos, e nos demais não passou de 7e-13. Nas notas da tabela seguinte, a maior diferença foi de 1,3e-14. Cinco desses cenários ficaram em `tests/models/test_disco.py`, com os valores do código oficial.

### Tempo e memória

Mesma partição (HDBSCAN com `min_cluster_size=50` e `min_samples=100`) e mesmas amostras estratificadas para todas as métricas. A memória é o pico do processo e dos seus filhos, incluindo cerca de 0,15 GB do próprio processo.

| Pontos | DBCV | DISCO oficial | DISCO deste repositório |
|---:|---:|---:|---:|
| 2.000 | 4,2 s / 0,70 GB | 0,3 s / 0,21 GB | 0,1 s / 0,16 GB |
| 5.000 | 5,7 s / 1,13 GB | 1,8 s / 0,52 GB | 0,5 s / 0,16 GB |
| 10.000 | 12,3 s / 2,66 GB | 7,1 s / 1,74 GB | 1,5 s / 0,16 GB |
| 15.000 | 24,0 s / 5,31 GB | 13,3 s / 3,70 GB | 3,2 s / 0,16 GB |
| 28.000 | 109,7 s / 9,76 GB | não medido | 9,8 s / 0,17 GB |
| 64.316 | não medido | não medido | 40,9 s / 0,20 GB |

O código oficial não foi medido a partir de 28.000 pontos porque as suas matrizes não caberiam nos 16 GB da máquina. Com 28.000 pontos, o DISCO foi cerca de 11 vezes mais rápido que o DBCV e usou cerca de 57 vezes menos memória. O DBCV da tabela usa 12 processos. Os tempos em GPU do HDBSCAN não entram nesta comparação.

### Custo por partição

Com a árvore pronta, o custo restante é pequeno e cresce com o número de agrupamentos. A árvore do fold levou 36,2 s, e as partições vieram do k-means, com 30% dos pontos marcados como ruído ao acaso.

| Agrupamentos | Tempo por partição |
|---:|---:|
| 3 | 0,03 s |
| 10 | 0,09 s |
| 30 | 0,26 s |

Essas partições arbitrárias ficaram em torno de -0,28, enquanto as do HDBSCAN ficaram entre 0,13 e 0,25, o que é coerente com o que o índice se propõe a medir.

### Comparação com o DBCV na grade

Grade do pipeline no mesmo fold, com `min_cluster_size=50`. O agrupamento veio do HDBSCAN do scikit-learn em CPU, e não do cuML, de modo que os valores podem diferir dos da execução no Colab.

| `min_samples` | Agrupamentos | Ruído | DISCO | DBCV |
|---:|---:|---:|---:|---:|
| 100 | 3 | 34,1% | 0,197 | 0,089 |
| 150 | 3 | 40,1% | 0,171 | 0,164 |
| 200 | 2 | 29,4% | **0,252** | 0,168 |
| 250 | 3 | 47,8% | 0,149 | **0,218** |
| 300 | 2 | 35,1% | 0,223 | 0,155 |
| 350 | 3 | 51,7% | 0,130 | 0,205 |
| 400 | 3 | 52,3% | 0,126 | 0,180 |

As duas métricas escolhem configurações diferentes neste fold: `min_samples=200` pelo DISCO e `min_samples=250` pelo DBCV. Nas sete combinações, a correlação de Pearson entre o DISCO e a fração de ruído foi de -0,97, e entre o DBCV e a fração de ruído, de 0,65. Ou seja, aqui o DISCO preferiu as partições com menos pontos de ruído. Sete combinações de um único fold não bastam para generalizar esse comportamento, e a tabela não mostra qual escolha detecta melhor as fraudes.

O custo total na grade também difere: o DBCV somou 570 s nas sete combinações, e o DISCO, 36 s da árvore e 0,4 s das partições.

## Limitações

- **Custo quadrático:** a construção da árvore continua O(n² d), em uma só thread na etapa de Prim. Na base completa (182 mil pontos), a projeção quadrática dá cerca de 5 minutos por fold, valor ainda não medido.
- **Média sobre todos os pontos:** como cada ponto pesa igual, uma fração grande de ruído influencia diretamente a nota.
- **Validação interna:** o índice mede a estrutura de densidade, não a detecção de fraudes. A relação entre o DISCO e a AUPRC ainda precisa ser avaliada nos resultados do pipeline.
- **Distância euclidiana:** a implementação cobre só essa distância, a mesma usada pelo HDBSCAN\* no pipeline.

## Testes

`tests/models/test_disco.py` confere as equações com exemplos calculados à mão, os casos especiais do artigo, os valores do código oficial e as distâncias da árvore contra a definição. `tests/models/test_pipeline_disco.py` confere a integração: a base subamostrada, a seleção pelo DISCO, a árvore única por fold e a recusa de retomar um histórico sem a métrica. Esses testes substituem o cuML e o dbcv por módulos vazios, de modo que não executam o pipeline de ponta a ponta.

```bash
uv run pytest tests/models/test_disco.py tests/models/test_pipeline_disco.py
```

## Reprodução

As tabelas vêm de [`src/analysis/disco_benchmark.py`](../../src/analysis/disco_benchmark.py), que usa um ambiente à parte porque o `dbcv` quebra com o SciPy 1.18. Na etapa `scaling`, `--original-dir` aponta para um clone do código oficial (usado o commit `e685d5b`).

```bash
uv run --no-project --python 3.13 --with numpy --with "scipy<1.17" --with scikit-learn \
    --with pandas --with psutil --with "git+https://github.com/FelSiq/DBCV" \
    python src/analysis/disco_benchmark.py {scaling,partitions,grid} [--original-dir <pasta>]
```

Os tempos variam com a máquina e a carga do momento.

## Referência

BEER, A.; KRIEGER, L.; WEBER, P.; RITZERT, M.; ASSENT, I.; PLANT, C. **Internal Evaluation of Density-Based Clusterings with Noise**. arXiv:2503.00127v2, 2025. PDF em [`docs/references/articles/density-based-clustering/cluster-validation/`](../references/articles/density-based-clustering/cluster-validation/beer2025_internal-evaluation-density-based-clusterings-noise.pdf).
