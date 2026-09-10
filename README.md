# Detecção de Fraudes Financeiras

Trabalho de Conclusão de Curso — Ciência da Computação, Universidade Federal de Lavras (UFLA).

Comparação entre o algoritmo não supervisionado de agrupamento por densidade **HDBSCAN** (fraude tratada como ruído) e as abordagens supervisionadas de classificação **Random Forest** e **XGBoost**, sobre a base de transações de cartão de crédito do Machine Learning Group da Université Libre de Bruxelles.

**Autor:** Heitor Rodrigues Sabino
**Orientadora:** Elaine Cecilia Gatto
**Coorientador:** Natanael Fabrício Dacioli Batista

---

## Conjunto de dados

`creditcard.csv` — 284.807 transações europeias de setembro de 2013, 31 variáveis, 492 fraudes (0,1727%). As variáveis `V1`–`V28` são componentes principais (PCA) anonimizados; `Time`, `Amount` e `Class` são originais.

O arquivo tem ~150 MB e **não é versionado** (acima do limite do GitHub). Baixe de
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
e coloque em `data/raw/creditcard.csv`.

---

## Estrutura do repositório

```
.
├── data/
│   ├── raw/                                  # base original (não versionada)
│   │   └── creditcard.csv
│   └── processed/                            # artefatos gerados (não versionados)
│       ├── outer_folds/                      # folds externos: treino/teste
│       └── inner_folds/                      # folds internos: treino/validação
│
├── src/
│   ├── preprocessing/
│   │   ├── cross_validation.py               # geração de folds estratificados
│   │   └── run_fold_generation.py            # entrada: constrói toda a árvore de folds
│   │
│   ├── models/
│   │   ├── grid_search.py                    # busca em grade (Random Forest / XGBoost)
│   │   ├── run_supervised_grid_search.py     # entrada: modelos supervisionados
│   │   └── run_hdbscan_grid_search.py        # entrada: HDBSCAN (grid por DBCV + avaliação externa)
│   │
│   └── utils/                                # vazia — reservada para utilitários
│
├── results/
│   ├── supervised/                           # saída de run_supervised_grid_search.py
│   └── unsupervised/                         # saída de run_hdbscan_grid_search.py
│       ├── hdbscan_grid_best_summary.csv
│       └── hdbscan_grid_results_by_fold.csv
│
├── tests/                                    # suíte de testes unitários
│   ├── conftest.py                           # fixtures: base sintética e folds
│   ├── preprocessing/
│   └── models/
│
├── docs/
│   ├── monografia/                           # texto do TCC (LaTeX)
│   │   ├── main.tex
│   │   ├── refbib.bib
│   │   └── uflamon.cls
│   └── references/                           # vault do Obsidian: bibliografia
│       ├── articles/                         # PDFs por tema (não versionados)
│       └── citations/                        # notas de leitura
│
├── pytest.ini
├── requirements.txt
└── README.md
```

**Convenções.** Todos os módulos usam `snake_case` em inglês. O prefixo `run_`
identifica os pontos de entrada executáveis; os demais arquivos são bibliotecas
importadas por eles. Os caminhos são relativos à raiz do repositório.

---

## Instalação

```bash
pip install -r requirements.txt
```

A etapa não supervisionada depende de `cuml-cu12` e `dbcv`, que exigem Linux com
GPU NVIDIA (CUDA 12) e não são instaláveis pelo `requirements.txt`. Os comandos
estão documentados no próprio arquivo. Esse experimento foi executado no Google Colab.

---

## Execução

Todos os scripts assumem a **raiz do repositório** como diretório de trabalho.

**1. Geração dos folds** — nested cross-validation estratificada, 5 folds externos 80/20 e divisão interna 80/20:

```bash
python src/preprocessing/run_fold_generation.py
```

Produz `data/processed/outer_folds/` (227.845 treino / 56.961 teste por fold) e
`data/processed/inner_folds/` (182.276 treino / 45.569 validação), com metadados em JSON.

**2. Grid search supervisionado** — Random Forest ou XGBoost, seleção por F1:

```bash
python src/models/run_supervised_grid_search.py
```

O modelo (`"rf"` ou `"xgb"`) e a grade são definidos no bloco `__main__`.
Resultados em `results/supervised/results_by_fold.csv` e `results_summary.csv`.

**3. HDBSCAN** — grid interno otimizado por DBCV e avaliação nos folds externos:

```bash
python src/models/run_hdbscan_grid_search.py
```

Requer GPU. As constantes `INTERNAL_FOLDS_DIR`, `OUTER_FOLDS_DIR` e `RESULTS_DIR`
no bloco `__main__` apontam para o Google Drive, refletindo o ambiente Colab em que
o experimento foi executado; ajuste-as para `data/processed/` e `results/unsupervised/`
ao rodar localmente.

---

## Ambiente de execução do HDBSCAN

O experimento não supervisionado foi executado no Google Colab. A máquina de
referência é a seguinte, com as especificações medidas na própria sessão:

| Recurso | Especificação |
|---|---|
| Acelerador | NVIDIA L4 |
| Memória da GPU | 23.034 MiB |
| Driver NVIDIA | 580.82.07 |
| Memória do sistema | 53,0 GiB |
| Processadores | 12 vCPUs (6 núcleos físicos) |
| Disco em `/content` | 113 GiB |
| cuML | 26.02 (pré-instalado na imagem) |
| scikit-learn / pandas / NumPy | 1.6.1 / 2.2.3 / 2.1.3 |
| `dbcv` | instalado a partir do repositório de origem |

**Por que a L4.** A escolha do acelerador é determinada pela memória, não pela
capacidade de cálculo. O ajuste do HDBSCAN leva segundos em qualquer GPU, ao
passo que o DBCV é executado em CPU e monta matrizes de distância de `n²` — para
os 28.000 pontos da amostra, 5,84 GiB por matriz, replicadas entre os processos.
O pico medido foi de **17,6 GiB**, o que exclui a faixa de 12,7 GiB da T4 e
dispensa os ~83 GiB da A100. A L4 é a menor máquina adequada, com folga de 2,7×.

### Tempo de execução

Medido em uma combinação (`min_cluster_size=5`, `min_samples=300`, fold interno 1)
e extrapolado para a grade completa:

| Etapa | Tempo |
|---|---|
| Ajuste do HDBSCAN | 16,5 s |
| Cálculo do DBCV (12 processos) | 159,1 s |
| **Por combinação** | **175,6 s** |
| **Grade de 21 combinações × 5 folds (105 execuções)** | **≈ 5,1 h** |

A extrapolação é aproximada: o custo do DBCV depende da estrutura de
agrupamento encontrada, e configurações que produzem mais clusters demoram mais.
Para uma única sessão de ambiente de execução, convém reservar o dobro da
estimativa — cerca de **12 horas**.

A reprodutibilidade foi verificada: a mesma combinação executada nesta máquina
reproduziu o resultado do experimento original até a sexta casa decimal
(DBCV 0,181026, 4 clusters, 54.220 pontos de ruído).

---

## Testes

```bash
pytest                  # suíte completa
pytest -m "not slow"    # exclui os testes que treinam modelos
pytest tests/preprocessing
```

A suíte cobre uma unidade por arquivo, espelhando a estrutura de `src/`:

| Arquivo | Unidade sob teste |
|---|---|
| `preprocessing/test_resolve_target_column.py` | resolução da coluna alvo por nome ou índice |
| `preprocessing/test_stratified_folds.py` | folds externos: exclusividade, cobertura e estratificação |
| `preprocessing/test_external_k_folds.py` | materialização dos folds externos e metadados |
| `preprocessing/test_internal_k_folds.py` | folds internos e isolamento do teste externo |
| `models/test_manual_grid_search.py` | busca em grade, agregação por combinação e seleção |

As fixtures constroem uma base sintética com a mesma estrutura de
`creditcard.csv` — 31 colunas, `Amount` assimétrica, `V1`–`V28` com variâncias
decrescentes e `Class` desbalanceada — de modo que a suíte roda sem o arquivo
de 150 MB, que não é versionado.

Os testes verificam as propriedades das quais a metodologia depende: cada
transação é testada exatamente uma vez ao longo dos folds, a proporção de
fraudes é preservada em cada partição, o conjunto de teste externo nunca
alcança as partições de calibração, o resumo por combinação corresponde à
média das métricas por fold e a execução paralela reproduz a sequencial.
`run_hdbscan_grid_search.py` fica fora da suíte por exigir GPU e por começar
com comandos de instalação do Colab, que impedem sua importação.

---

## Monografia

O texto está em `docs/monografia/main.tex`, na classe `uflamon`, com referências
em ABNT via `abntex2cite`. A compilação é feita no
[Overleaf](https://www.overleaf.com/), e a versão mais recente do PDF é
mantida versionada em `docs/monografia/main.pdf`.

A classe exige o logo da UFLA em `docs/monografia/logoufla.pdf` (ou `.png`),
referenciado por `\includegraphics{logoufla}` na capa.

---

## Metodologia

- **Validação:** nested cross-validation estratificada — 5 folds externos (80/20) e, dentro de cada treino externo, uma divisão interna 80/20 para calibração. As partições de seleção de hiperparâmetros nunca coincidem com as de estimativa de desempenho.
- **Pré-processamento:** `Time` padronizado por Z-score (natureza incremental e limitada); `Amount` por *Robust Scaler* (assimetria de 16,98 e curtose de 845 — mediana e IQR evitam distorção pelos valores extremos); `V1`–`V28` mantidos, por já resultarem de PCA e serem mutuamente ortogonais.
- **Otimização:** busca em grade exaustiva no ciclo interno — por F1 nos modelos supervisionados, por DBCV (*Density-Based Clustering Validation*) no HDBSCAN, mantendo este último inteiramente não supervisionado na seleção.
