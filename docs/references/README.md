# Referências

Vault do Obsidian com a bibliografia do trabalho. `articles/` guarda os PDFs,
agrupados por tema; `citations/` guarda as notas de leitura, que apontam para
trechos específicos dos PDFs por meio de wikilinks.

## Notas de leitura

As notas seguem a estrutura do `main.tex`; o prefixo numérico preserva a ordem
dos capítulos na listagem do Obsidian.

| Nota | Seção correspondente na monografia |
|---|---|
| `01-theoretical-background/machine-learning.md` | 3.1 Aprendizado de Máquina |
| `01-theoretical-background/principal-component-analysis.md` | 3.1.3.2 Principal Component Analysis |
| `01-theoretical-background/scaling-and-standardization.md` | 3.1.3.3 Escalonamento e padronização |
| `01-theoretical-background/nested-cross-validation.md` | 3.1.3.4 Validação Cruzada Aninhada |
| `01-theoretical-background/xgboost.md` | 3.2.2 Extreme Gradient Boosting |
| `01-theoretical-background/anomaly-detection.md` | 3.3 Algoritmo Não Supervisionado |
| `02-methodology/grid-search.md` | 4.3.2 Grid Search |
| `02-methodology/hyperparameter-selection.md` | 4.3 Parametrização |
| `02-methodology/density-based-clustering-validation.md` | 4.3 Parametrização — critério DBCV |
| `03-support/bibtex-references.md` | entradas BibTeX coletadas |
| `03-support/ideas.md` | anotações de trabalho e pontos em aberto |

## Convenção de nomes

```
<primeiroautor><ano>_<titulo-curto>.pdf
```

A raiz do nome é a mesma chave usada no `refbib.bib`, de modo que um `\cite{schubert2017}`
no `main.tex` leve direto ao arquivo `schubert2017_dbscan-revisited.pdf`.

Os wikilinks do Obsidian resolvem por **nome de arquivo**, não por caminho — mover
um PDF entre subpastas não quebra as notas, mas renomeá-lo quebra.

## Acervo

A coluna **bib** indica que a chave já consta no `docs/monografia/refbib.bib`.
Atualmente as 27 entradas do `refbib.bib` estão todas citadas no `main.tex`, de
modo que ✓ equivale a "em uso". Cada nota de `citations/` traz no topo um bloco
de status com essa informação e a seção onde a obra é citada.

**Lidos mas ainda não incorporados:** `samariya2023`, `moulavi2014` e
`alcino2022` têm notas de leitura mas nenhuma entrada no `refbib.bib`.
`moulavi2014` é a fonte original do DBCV, critério de seleção do grid do
HDBSCAN — usado no método, mas ainda sem citação no texto.

### fraud-detection

| Chave | bib | Referência |
|---|:-:|---|
| `alarfaj2022` | ✓ | Alarfaj et al. *Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms*. IEEE Access, 2022. |
| `sulaiman2022` | | Sulaiman, Schetinin & Sant. *Review of Machine Learning Approach on Credit Card Fraud Detection*. Human-Centric Intelligent Systems, 2022. |
| `jurgovsky2018` | | Jurgovsky et al. *Sequence Classification for Credit-Card Fraud Detection*. Expert Systems with Applications, 2018. |
| `ghalwash2025` | | Ghalwash et al. *Enhancing credit card fraud detection using DBSCAN-augmented disjunctive voting ensemble*. Scientific Reports, 2025. |
| `santos2025` | | Santos, Bueno dos Santos & Toma. *Detecção de transações fraudulentas com cartão de crédito: uma abordagem comparativa baseada em aprendizado de máquina*. Revista Foco, 2025. |
| `alcino2022` | | Alcino. *Modelos de classificação em fraudes financeiras: comparação de desempenho em casos de crime de smurfing*. Dissertação, UFLA, 2022. |

### machine-learning-foundations

| Chave | bib | Referência |
|---|:-:|---|
| `rezazadeh2025` | ✓ | Rezazadeh. *Review of Machine Learning*. UJRRA, 2025. |

### supervised-models

| Chave | bib | Referência |
|---|:-:|---|
| `chen2016` | ✓ | Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System*. KDD, 2016. |
| `biau2016` | | Biau & Scornet. *A random forest guided tour*. TEST, 2016. |
| `probst2019` | | Probst, Wright & Boulesteix. *Hyperparameters and tuning strategies for random forest*. WIREs Data Mining and Knowledge Discovery, 2019. |

### density-based-clustering

| Chave | bib | Referência |
|---|:-:|---|
| `schubert2017` | ✓ | Schubert, Sander, Ester, Kriegel & Xu. *DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN*. ACM TODS, 2017. |
| `campello2013` | | Campello, Moulavi & Sander. *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD, 2013. — artigo original do HDBSCAN. |
| `campello2015` | | Campello, Moulavi, Zimek & Sander. *Hierarchical density estimates for data clustering, visualization, and outlier detection*. ACM TKDD, 2015. |
| `moulavi2014` | | Moulavi, Jaskowiak, Campello, Zimek & Sander. *Density-Based Clustering Validation*. SDM, 2014. — origem do índice DBCV, critério de seleção do grid do HDBSCAN. |
| `akbari2016` | | Akbari & Unland. *Automated Determination of the Input Parameter of DBSCAN Based on Outlier Detection*. AIAI, 2016. |
| `frenzel2021` | | Frenzel. *How To Tune HDBSCAN*. Towards Data Science, 2021. — fonte não revisada por pares; usar como apoio prático, não como citação de método. |
| `thompson2024` | | Thompson. *Partition-Free Cluster Evaluation: Extending Cluster Validation and Cluster Extraction from Hierarchies*. Dissertação de mestrado, University of Alberta, 2024. — propõe o PFCE, candidato a substituir o DBCV na seleção do grid. |
| `neto2022` | | Neto, Naldi, Campello & Sander. *CORE-SG: Efficient Computation of Multiple MSTs for Density-Based Methods*. ICDE, 2022. |

### outlier-detection

| Chave | bib | Referência |
|---|:-:|---|
| `samariya2023` | | Samariya. *A Comprehensive Survey of Anomaly Detection Algorithms*. 2023. |
| `boukerche2020` | | Boukerche, Zheng & Alfandi. *Outlier Detection: Methods, Models, and Classification*. ACM Computing Surveys, 2020. |
| `ghosh2024` | | Ghosh, Naldi, Sander & Choo. *Unsupervised Parameter-free Outlier Detection using HDBSCAN* Outlier Profiles*. 2024. |

### preprocessing

| Chave | bib | Referência |
|---|:-:|---|
| `abdi2010` | ✓ | Abdi & Williams. *Principal component analysis*. WIREs Computational Statistics, 2010. |
| `greenacre2022` | ✓ | Greenacre et al. *Principal component analysis*. Nature Reviews Methods Primers, 2022. |
| `huang2015` | ✓ | Huang et al. *An empirical analysis of data preprocessing for machine learning-based software cost estimation*. 2015. |
| `vandermaaten2008` | | van der Maaten & Hinton. *Visualizing Data using t-SNE*. JMLR, 2008. |
| `balogun2019` | | Balogun et al. *Performance Analysis of Feature Selection Methods in Software Defect Prediction*. Applied Sciences, 2019. |
| `lima2017` | | Lima & Pereira. *Feature Selection Approaches to Fraud Detection in e-Payment Systems*. ICCSA, 2017. |

### model-validation

| Chave | bib | Referência |
|---|:-:|---|
| `krstajic2014` | ✓ | Krstajic, Buturovic, Leahy & Thomas. *Cross-validation pitfalls when selecting and assessing regression and classification models*. Journal of Cheminformatics, 2014. |
| `liashchynskyi2019` | ✓ | Liashchynskyi & Liashchynskyi. *Grid Search, Random Search, Genetic Algorithm: A Big Comparison for NAS*. 2019. |
| `shams2024` | ✓ | Shams et al. *Water quality prediction using machine learning models based on grid search method*. 2024. |
| `berrar2019` | | Berrar. *Cross-validation*. Encyclopedia of Bioinformatics and Computational Biology, 2019. — **ano a confirmar**: o PDF não o traz. |

### figures

Figuras de trabalho, geradas durante a análise: `auc-variation.png` e
`base-umap-visualization.png`.

## Versionamento

Os PDFs (`articles/`) somam cerca de 48 MB e são material de editoras, cuja
redistribuição em geral não é permitida — por isso não são versionados. O que
vai para o repositório são as notas de `citations/`, este índice e o
`refbib.bib`. A configuração local do Obsidian (`.obsidian/`) também fica fora.
