# Referências

Vault do Obsidian com a bibliografia do trabalho. `articles/` guarda os PDFs,
agrupados por tema e, dentro de cada tema, por algoritmo ou técnica;
`citations/` guarda as notas de leitura, que apontam para trechos específicos
dos PDFs por meio de wikilinks.

## Notas de leitura

As notas seguem a estrutura do `main.tex`; o prefixo numérico preserva a ordem
dos capítulos na listagem do Obsidian.

| Nota | Seção correspondente na monografia |
|---|---|
| `citation-summary.md` | sumário de todas as citações do `main.tex`, em ordem, com links para os grifos |
| `00-introduction/introduction.md` | 1 Introdução |
| `00-related-work/related-work.md` | 2 Trabalhos Correlatos |
| `01-theoretical-background/machine-learning.md` | 3.1 Aprendizado de Máquina |
| `01-theoretical-background/principal-component-analysis.md` | 3.1.3.1 Principal Component Analysis |
| `01-theoretical-background/scaling-and-standardization.md` | 3.1.3.2 Escalonamento e padronização |
| `01-theoretical-background/nested-cross-validation.md` | 3.1.3.3 Validação Cruzada Aninhada |
| `01-theoretical-background/random-forest.md` | 3.2.1 Random Forest |
| `01-theoretical-background/xgboost.md` | 3.2.2 Extreme Gradient Boosting |
| `01-theoretical-background/anomaly-detection.md` | 3.3 Algoritmo Não Supervisionado |
| `01-theoretical-background/dbscan.md` | 3.3.1 DBSCAN* |
| `01-theoretical-background/hdbscan.md` | 3.3.2 HDBSCAN* |
| `01-theoretical-background/mahalanobis-distance.md` | estudo da distância de Mahalanobis, ainda sem seção no `main.tex` |
| `01-theoretical-background/evaluation-metrics.md` | 3.4 Métricas de Avaliação |
| `02-methodology/preprocessing.md` | 4.2 Pré-processamento |
| `02-methodology/grid-search.md` | 4.3.2 Grid Search |
| `02-methodology/hyperparameter-selection.md` | 4.3 Parametrização |
| `02-methodology/density-based-clustering-validation.md` | 4.3 Parametrização — critério DBCV |
| `02-methodology/hyperparameter-grids.md` | 4.3.3 Grades de Hiperparâmetros |
| `02-methodology/fraud-detection.md` | 4.4 Detecção de Fraudes |
| `02-methodology/execution-environment.md` | 4.5 Ambiente de Execução e Bibliotecas |
| `03-conclusion/conclusion.md` | 6 Conclusão |
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
Atualmente as 64 entradas do `refbib.bib` estão todas citadas no `main.tex`, de
modo que ✓ equivale a "em uso". Cada nota de `citations/` traz no topo um bloco
de status com essa informação e a seção onde a obra é citada.

**Lidos mas ainda não incorporados:** `samariya2023`, `alcino2022`,
`demaesschalck2000`, `todeschini2013`, `evans2001`, `xie2024`, `bulut2019`,
`jain1999` e `oyewole2023` têm notas de leitura mas nenhuma entrada no
`refbib.bib`.

### fraud-detection

| Chave | bib | Referência |
|---|:-:|---|
| `alarfaj2022` | ✓ | Alarfaj et al. *Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms*. IEEE Access, 2022. |
| `sulaiman2022` | ✓ | Sulaiman, Schetinin & Sant. *Review of Machine Learning Approach on Credit Card Fraud Detection*. Human-Centric Intelligent Systems, 2022. |
| `jurgovsky2018` | ✓ | Jurgovsky et al. *Sequence Classification for Credit-Card Fraud Detection*. Expert Systems with Applications, 2018. |
| `ghalwash2025` | ✓ | Ghalwash et al. *Enhancing credit card fraud detection using DBSCAN-augmented disjunctive voting ensemble*. Scientific Reports, 2025. |
| `hajek2022` | ✓ | Hajek, Abedin & Sivarajah. *Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework*. Information Systems Frontiers, 2022. O PDF é a versão on-line; no fascículo, v. 25, n. 5, p. 1985 a 2003, 2023. |
| `ngai2011` | ✓ | Ngai, Hu, Wong, Chen & Sun. *The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature*. Decision Support Systems, 2011. |
| `west2016` | ✓ | West & Bhattacharya. *Intelligent financial fraud detection: A comprehensive review*. Computers & Security, 2016. O PDF é a versão aceita do manuscrito. |
| `dalpozzolo2018` | ✓ | Dal Pozzolo, Boracchi, Caelen, Alippi & Bontempi. *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*. IEEE Transactions on Neural Networks and Learning Systems, 2018. |
| `santos2025` | | Santos, Bueno dos Santos & Toma. *Detecção de transações fraudulentas com cartão de crédito: uma abordagem comparativa baseada em aprendizado de máquina*. Revista Foco, 2025. |
| `alcino2022` | | Alcino. *Modelos de classificação em fraudes financeiras: comparação de desempenho em casos de crime de smurfing*. Dissertação, UFLA, 2022. |
| `xie2024` | | Xie & Huang. *A Credit Card Fraud Detection Method Based on Mahalanobis Distance Hybrid Sampling and Random Forest Algorithm*. IEEE Access, v. 12, p. 162788 a 162798, 2024. Usa a mesma base de cartões, mas reamostra antes de dividir treino e teste, e as métricas iguais a 1 não servem de comparação. |

### machine-learning-foundations

| Chave | bib | Referência |
|---|:-:|---|
| `rezazadeh2025` | ✓ | Rezazadeh. *Review of Machine Learning*. UJRRA, 2025. |
| `jain1999` | | Jain, Murty & Flynn. *Data Clustering: A Review*. ACM Computing Surveys, 1999. Sustentava, por apud de Oyewole e Thopil (2023), a pertinência ao agrupamento, que o grau de anomalia não usa mais. |
| `oyewole2023` | | Oyewole & Thopil. *Data clustering: application and trends*. Artificial Intelligence Review, 2023. |

### supervised-models

#### random-forest

| Chave | bib | Referência |
|---|:-:|---|
| `biau2016` | ✓ | Biau & Scornet. *A random forest guided tour*. TEST, 2016. |
| `probst2019` | ✓ | Probst, Wright & Boulesteix. *Hyperparameters and tuning strategies for random forest*. WIREs Data Mining and Knowledge Discovery, 2019. |
| `chen2004` | ✓ | Chen, Liaw & Breiman. *Using Random Forest to Learn Imbalanced Data*. Relatório técnico 666, Departamento de Estatística, UC Berkeley, 2004. |

#### xgboost

| Chave | bib | Referência |
|---|:-:|---|
| `chen2016` | ✓ | Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System*. KDD, 2016. |

### density-based-clustering

#### dbscan

| Chave | bib | Referência |
|---|:-:|---|
| `schubert2017` | ✓ | Schubert, Sander, Ester, Kriegel & Xu. *DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN*. ACM TODS, 2017. |
| `akbari2016` | | Akbari & Unland. *Automated Determination of the Input Parameter of DBSCAN Based on Outlier Detection*. AIAI, 2016. |
| `ester1996` | ✓ | Ester, Kriegel, Sander & Xu. *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*. KDD, 1996. Artigo original do DBSCAN. |
| `sander1998` | ✓ | Sander, Ester, Kriegel & Xu. *Density-Based Clustering in Spatial Databases: The Algorithm GDBSCAN and Its Applications*. Data Mining and Knowledge Discovery, 1998. Sem PDF; citado por apud de Schubert et al. (2017). |

#### optics

| Chave | bib | Referência |
|---|:-:|---|
| `ankerst1999` | | Ankerst, Breunig, Kriegel & Sander. *OPTICS: Ordering Points To Identify the Clustering Structure*. ACM SIGMOD Record, 1999. |

#### hdbscan

| Chave | bib | Referência |
|---|:-:|---|
| `campello2013` | | Campello, Moulavi & Sander. *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD, 2013. Artigo original do HDBSCAN. |
| `campello2013b` | ✓ | Campello, Moulavi, Zimek & Sander. *A framework for semi-supervised and unsupervised optimal extraction of clusters from hierarchies*. Data Mining and Knowledge Discovery, 2013. |
| `campello2015` | ✓ | Campello, Moulavi, Zimek & Sander. *Hierarchical density estimates for data clustering, visualization, and outlier detection*. ACM TKDD, 2015. |
| `neto2021` | ✓ | Neto, Sander, Campello & Nascimento. *Efficient Computation and Visualization of Multiple Density-Based Clustering Hierarchies*. IEEE TKDE, 2021. |
| `mcinnes2017` | | McInnes & Healy. *Accelerated Hierarchical Density Based Clustering*. IEEE ICDM Workshops, 2017. |
| `neto2022` | ✓ | Neto, Naldi, Campello & Sander. *CORE-SG: Efficient Computation of Multiple MSTs for Density-Based Methods*. ICDE, 2022. |
| `ghosh2024` | ✓ | Ghosh, Naldi, Sander & Choo. *Unsupervised Parameter-free Outlier Detection using HDBSCAN* Outlier Profiles*. 2024. |
| `frenzel2021` | | Frenzel. *How To Tune HDBSCAN*. Towards Data Science, 2021. Fonte não revisada por pares; usar como apoio prático, não como citação de método. |

#### cluster-validation

| Chave | bib | Referência |
|---|:-:|---|
| `moulavi2014` | ✓ | Moulavi, Jaskowiak, Campello, Zimek & Sander. *Density-Based Clustering Validation*. SDM, 2014. Origem do índice DBCV, critério de seleção do grid do HDBSCAN. |
| `thompson2024` | | Thompson. *Partition-Free Cluster Evaluation: Extending Cluster Validation and Cluster Extraction from Hierarchies*. Dissertação de mestrado, University of Alberta, 2024. Propõe o PFCE, candidato a substituir o DBCV na seleção do grid. |
| `beer2025` | ✓ | Beer, Krieger, Weber, Ritzert, Assent & Plant. *Internal Evaluation of Density-Based Clusterings with Noise*. arXiv:2503.00127, 2025. Propõe o DISCO, índice interno que avalia também a qualidade da atribuição de ruído; código em https://github.com/pasiweber/DISCO. Critério de seleção do pipeline modular do HDBSCAN\* aprimorado, documentado em `docs/metrics/disco.md`. |

### outlier-detection

| Chave | bib | Referência |
|---|:-:|---|
| `samariya2023` | | Samariya. *A Comprehensive Survey of Anomaly Detection Algorithms*. 2023. |
| `boukerche2020` | | Boukerche, Zheng & Alfandi. *Outlier Detection: Methods, Models, and Classification*. ACM Computing Surveys, 2020. |
| `ramaswamy2000` | ✓ | Ramaswamy, Rastogi & Shim. *Efficient Algorithms for Mining Outliers from Large Data Sets*. ACM SIGMOD, 2000. Origem do escore de anomalia pela distância ao k-ésimo vizinho. |

#### mahalanobis-distance

| Chave | bib | Referência |
|---|:-:|---|
| `ghorbani2019` | ✓ | Ghorbani. *Mahalanobis Distance and Its Application for Detecting Multivariate Outliers*. Facta Universitatis, Series: Mathematics and Informatics, v. 34, n. 3, p. 583 a 595, 2019. |
| `mahalanobis1936` | ✓ | Mahalanobis. *On the Generalized Distance in Statistics*. Proceedings of the National Institute of Sciences (Calcutta), v. 2, p. 49 a 55, 1936. Sem PDF; citado por apud de Ghorbani (2019). |
| `hubert2010` | ✓ | Hubert & Debruyne. *Minimum Covariance Determinant*. WIREs Computational Statistics, v. 2, n. 1, p. 36 a 43, 2010. Revisão do MCD: definição, repesagem, distância robusta e FAST-MCD. O PDF traz a data de 2009 do copyright, mas o volume é de 2010. |
| `fauconnier2009` | ✓ | Fauconnier & Haesbroeck. *Outliers Detection with the Minimum Covariance Determinant Estimator in Practice*. Statistical Methodology, v. 6, n. 4, p. 363 a 379, 2009. Variância generalizada e repesagem com pesos 0 e 1. |
| `rousseeuw1985` | ✓ | Rousseeuw. *Multivariate Estimation with High Breakdown Point*. In: Grossmann, Pflug, Vincze & Wertz (org.). *Mathematical Statistics and Applications*, v. B. Reidel, 1985, p. 283 a 297. Sem PDF; citado por apud de Ghorbani (2019). |
| `rousseeuw1999` | ✓ | Rousseeuw & Van Driessen. *A Fast Algorithm for the Minimum Covariance Determinant Estimator*. Technometrics, v. 41, n. 3, p. 212 a 223, 1999. Sem PDF; citado por apud de Hubert e Debruyne (2010). É o FAST-MCD que o `MinCovDet` do scikit-learn implementa. |
| `bulut2019` | | Bulut & Zaman. *An Improved Class of Robust Ratio Estimators by Using the Minimum Covariance Determinant Estimation*. Communications in Statistics: Simulation and Computation, 2019. Atribui a definição do MCD a Rousseeuw (1985) e não acrescenta à revisão de Hubert e Debruyne (2010). |
| `demaesschalck2000` | | De Maesschalck, Jouan-Rimbaud & Massart. *The Mahalanobis distance*. Chemometrics and Intelligent Laboratory Systems, v. 50, p. 1 a 18, 2000. Tutorial sobre as distâncias euclidiana e de Mahalanobis no espaço original e no dos componentes principais. |
| `todeschini2013` | | Todeschini, Ballabio, Consonni, Sahigara & Filzmoser. *Locally centred Mahalanobis distance: A new distance measure with salient features towards outlier detection*. Analytica Chimica Acta, v. 787, p. 1 a 9, 2013. |
| `evans2001` | | Evans. *Statistical aspects of the detection of fraud*. In: Lock, Wells & Farthing (org.). *Fraud and Misconduct in Biomedical Research*. 3. ed. BMJ Books, 2001, cap. 14, p. 186 a 206. O PDF é o livro inteiro: a numeração do PDF coincide com a do livro até a p. 198 e fica uma página atrás depois dela. |
| `laca` | | Laca, Rubiano, Sulit, Magon, Concepcion & Bandala. *Runtime Optimization of Mahalanobis Distance for Anomalies in Credit Card Fraud Detection Dataset*. Sem PDF: o `.txt` traz o resumo e o começo da introdução. Ano e veículo não localizados. Renomear com o ano quando o artigo for obtido. |

### preprocessing

#### pca

| Chave | bib | Referência |
|---|:-:|---|
| `abdi2010` | ✓ | Abdi & Williams. *Principal component analysis*. WIREs Computational Statistics, 2010. |
| `greenacre2022` | ✓ | Greenacre et al. *Principal component analysis*. Nature Reviews Methods Primers, 2022. |

#### t-sne

| Chave | bib | Referência |
|---|:-:|---|
| `vandermaaten2008` | | van der Maaten & Hinton. *Visualizing Data using t-SNE*. JMLR, 2008. |

#### feature-selection

| Chave | bib | Referência |
|---|:-:|---|
| `balogun2019` | | Balogun et al. *Performance Analysis of Feature Selection Methods in Software Defect Prediction*. Applied Sciences, 2019. |
| `lima2017` | | Lima & Pereira. *Feature Selection Approaches to Fraud Detection in e-Payment Systems*. ICCSA, 2017. |

#### scaling

| Chave | bib | Referência |
|---|:-:|---|
| `huang2015` | ✓ | Huang et al. *An empirical analysis of data preprocessing for machine learning-based software cost estimation*. 2015. |

#### undersampling

| Chave | bib | Referência |
|---|:-:|---|
| `dalpozzolo2015` | ✓ | Dal Pozzolo, Caelen, Johnson & Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium Series on Computational Intelligence (SSCI), 2015. Artigo de origem do dataset (p. 6), base da subamostragem e dos pesos das legítimas na base subamostrada, inclusive no limiar de alerta. O PDF é a versão final dos autores, sem a numeração dos anais. |

### statistics

| Chave | bib | Referência |
|---|:-:|---|
| `joanes1998` | ✓ | Joanes & Gill. *Comparing measures of sample skewness and kurtosis*. Journal of the Royal Statistical Society: Series D (The Statistician), 1998. Definição do coeficiente de assimetria amostral $g_1$. |

### model-validation

#### cross-validation

| Chave | bib | Referência |
|---|:-:|---|
| `krstajic2014` | ✓ | Krstajic, Buturovic, Leahy & Thomas. *Cross-validation pitfalls when selecting and assessing regression and classification models*. Journal of Cheminformatics, 2014. |
| `berrar2024` | ✓ | Berrar. *Cross-validation*. Preprint da versão revisada para a 2ª edição da Encyclopedia of Bioinformatics and Computational Biology (Elsevier), 2024. A 1ª edição é de 2018, p. 542 a 545. |

#### evaluation-metrics

| Chave | bib | Referência |
|---|:-:|---|
| `fawcett2006` | ✓ | Fawcett. *An introduction to ROC analysis*. Pattern Recognition Letters, 2006. Matriz de confusão e métricas derivadas dela. |
| `chawla2002` | ✓ | Chawla, Bowyer, Hall & Kegelmeyer. *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 2002. Acurácia inadequada em bases desbalanceadas. |
| `davis2006` | ✓ | Davis & Goadrich. *The relationship between Precision-Recall and ROC curves*. ICML, 2006. |
| `campos2016` | ✓ | Campos, Zimek, Sander, Campello, Micenková, Schubert, Assent & Houle. *On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study*. Data Mining and Knowledge Discovery, 2016. |
| `zhang2009` | ✓ | Zhang & Zhang. *Average Precision*. Encyclopedia of Database Systems, Springer, 2009. O PDF reúne todos os verbetes da letra A, e o verbete está na p. 192. |
| `saito2015` | | Saito & Rehmsmeier. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 2015. |

#### grid-search

| Chave | bib | Referência |
|---|:-:|---|
| `liashchynskyi2019` | ✓ | Liashchynskyi & Liashchynskyi. *Grid Search, Random Search, Genetic Algorithm: A Big Comparison for NAS*. 2019. |
| `shams2024` | ✓ | Shams et al. *Water quality prediction using machine learning models based on grid search method*. 2024. |

### software

| Chave | bib | Referência |
|---|:-:|---|
| `pedregosa2011` | ✓ | Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011. |
| `raschka2020` | ✓ | Raschka, Patterson & Nolet. *Machine Learning in Python: Main Developments and Technology Trends in Data Science, Machine Learning, and Artificial Intelligence*. Information, 2020. RAPIDS e cuML. |

### Documentação (sem PDF)

As entradas abaixo são páginas de documentação das bibliotecas, conferidas em
23/10/2026. Os trechos citados ficam nas notas, com o link da página.

| Chave | bib | Referência |
|---|:-:|---|
| `sklearn2026rf` | ✓ | Scikit-learn Developers. *sklearn.ensemble.RandomForestClassifier*. |
| `xgboost2026` | ✓ | XGBoost Developers. *XGBoost Parameters*. |
| `cuml2026` | | NVIDIA. *cuML API Reference*, entradas HDBSCAN e approximate_predict. Sustentava a pertinência e a atribuição aproximada, que o grau de anomalia não usa mais. |

### figures

Figuras de trabalho, geradas durante a análise: `auc-variation.png` e
`base-umap-visualization.png`.

## Versionamento

Os PDFs (`articles/`) somam cerca de 91 MB e são material de editoras, cuja
redistribuição em geral não é permitida, por isso não são versionados. O mesmo
vale para os resumos copiados em `.txt` quando o PDF não está disponível. O que
vai para o repositório são as notas de `citations/`, este índice e o
`refbib.bib`. A configuração local do Obsidian (`.obsidian/`) também fica fora.
