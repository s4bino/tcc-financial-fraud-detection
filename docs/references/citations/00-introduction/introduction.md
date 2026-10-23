> [!check] Status: em uso na monografia
> Todas as chaves constam do `refbib.bib` e são citadas na Introdução.
> - `ngai2011`: categorias de fraude financeira, fraude com cartão entre as bancárias, classes de técnicas de mineração de dados e técnicas mais usadas. Citado pelo Alcino (2022) e por Jurgovsky et al. (2018).
> - `alarfaj2022`: definição da fraude com cartão, fraude sem a presença do cartão e perdas. Também citado em Trabalhos Correlatos e em Metodologia › Conjunto de Dados.
> - `jurgovsky2018`: detecção como preocupação dos sistemas bancários, sistema típico de detecção, regras de especialistas e de ML, abordagem supervisionada, detecção de anomalias e fraudes misturadas às transações legítimas. Também citado em Trabalhos Correlatos.
> - `west2016`: detecção manual impraticável e adoção de métodos estatísticos e computacionais. Citado pelo Alcino (2022) para a mesma afirmação. Também citado em Fundamentação Teórica › Random Forest.
> - `dalpozzolo2018`: desafios da detecção e rótulos fornecidos pelos investigadores, em geral com atraso. Citado por Jurgovsky et al. (2018) como 2017, ano da publicação antecipada.
> - `sulaiman2022`, `ghosh2024`, `campello2015`, `moulavi2014` e `davis2006`: restrição de acesso a dados reais, premissa da baixa densidade, HDBSCAN\* com agrupamento e detecção de *outliers*, DBCV sem rótulos e AUPRC sob desbalanceamento. Os grifos de Sulaiman, Ghosh, Moulavi e Davis repetem os das notas de Trabalhos Correlatos e da Fundamentação Teórica.
> - `kaggle2018`: período, origem e proporção de fraudes da base. É a página do conjunto de dados no Kaggle, sem PDF.

## Fraude financeira e fraude com cartão de crédito

> [!PDF|yellow] [[ngai2011_data-mining-financial-fraud-detection-review.pdf#page=1&selection=71,68,84,82&color=yellow|ngai2011_data-mining-financial-fraud-detection-review, p.1]]
> > 49 journal articles on the subject published between 1997 and 2008 was analyzed and classified into four categories of financial fraud (bank fraud, insurance fraud, securities and commodities fraud, and other related financial fraud) and six classes of data mining techniques (classification, regression, clustering, prediction, outlier detection, and visualization).
>
> Página 559 da revista. Sustenta as quatro categorias de fraude financeira do primeiro parágrafo e as seis classes de técnicas do quarto.

^ngai2011-p1

> [!PDF|yellow] [[ngai2011_data-mining-financial-fraud-detection-review.pdf#page=3&selection=275,0,294,31&color=yellow|ngai2011_data-mining-financial-fraud-detection-review, p.3]]
> > In this study, bank fraud includes credit card fraud, money laundering, and mortgage fraud,
>
> Página 561 da revista. Sustenta que a fraude com cartão de crédito figura entre as fraudes bancárias.

^ngai2011-p3

> [!PDF|yellow] [[alarfaj2022_credit-card-fraud-detection-state-of-the-art.pdf#page=1&selection=71,0,74,60&color=yellow|alarfaj2022_credit-card-fraud-detection-state-of-the-art, p.1]]
> > Credit card fraud (CCF) is a type of identity theft in which someone other than the owner makes an unlawful transaction using a credit card or account details. A credit card that has been stolen, lost, or counterfeited might result in
>
> Página 39700 da revista. Sustenta a definição da fraude com cartão e a sua origem em cartões roubados, perdidos ou falsificados. A frase termina com "fraud." na coluna seguinte.

^alarfaj2022-p1

## Pagamentos eletrônicos e perdas

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=5&selection=6,0,10,75&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.5]]
> > As businesses continue to evolve and migrate to the internet and money is transacted electronically in an ever-growing cashless banking economy, accurate fraud detection remains a key concern for modern banking systems. It is not only to limit the direct losses incurred by fraudulent transactions, but also to ensure that legitimate customers are not adversely impacted by automated and manual reviews.
>
> Página do PDF local, que é a versão aceita do manuscrito. Sustenta a detecção como preocupação central dos sistemas bancários, as perdas diretas e o efeito das revisões sobre os clientes legítimos.

^jurgovsky2018-p5

> [!PDF|yellow] [[alarfaj2022_credit-card-fraud-detection-state-of-the-art.pdf#page=1&selection=81,7,86,50&color=yellow|alarfaj2022_credit-card-fraud-detection-state-of-the-art, p.1]]
> > Card-not-present fraud, or the use of your credit card number in e-commerce transactions has also become increasingly common as a result of the increase in online shopping. Increased fraud, such as CCF, has resulted from the expansion of e-banking and several online payment environments, resulting in annual losses of billions of dollars.
>
> Página 39700 da revista. Sustenta a fraude sem a presença do cartão e as perdas anuais de bilhões de dólares.

^alarfaj2022-p1-2

## Detecção manual e sistema de detecção

> [!PDF|yellow] [[west2016_intelligent-financial-fraud-detection-review.pdf#page=4&selection=14,44,18,8&color=yellow|west2016_intelligent-financial-fraud-detection-review, p.4]]
> > Traditional methods involving manual detection are not only time consuming, expensive and inaccurate, but in the age of big data they are also impractical. Not surprisingly, financial institutions have turned to automated processes using statistical and computational methods.
>
> Página do PDF local, que é a versão aceita do manuscrito (resumo). Sustenta a detecção manual lenta, cara, imprecisa e impraticável e a adoção de processos automatizados.

^west2016-p4

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=5&selection=14,0,33,9&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.5]]
> > A typical fraud detection systems is composed of an automatic tool and a manual process. The automatic tool is based on fraud detection rules. It analyzes all the new incoming transactions and assigns a fraudulent score. The manual process is made by fraud investigators. They focus on transactions with high fraudulent score and provide a binary feedback (fraud or genuine) on all the transactions they analyzed.
>
> Sustenta o sistema típico de detecção: ferramenta automática que atribui um escore e investigadores que examinam as transações de maior escore.

^jurgovsky2018-p5-2

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=5&selection=34,0,62,43&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.5]]
> > The fraud detection systems can be based on expert driven rules, data driven rules or a combination of both types of rules. Expert driven rules try to identify specific scenarios of fraud discovered by the fraud investigators. A scenario of fraud can be "a cardholder does a transaction in a given country and, in the two next weeks, (s)he does another transaction for a given amount in another given country." If this scenario is detected in the stream of transactions then the fraud detection system will produce an alert. Data driven rules are based on machine learning algorithms. They learn the fraudulent patterns and try to detect them in a data-stream of new incoming transactions.
>
> Sustenta as regras definidas por especialistas e as aprendidas por algoritmos de ML.

^jurgovsky2018-p5-3

## Técnicas de mineração de dados

> [!PDF|yellow] [[ngai2011_data-mining-financial-fraud-detection-review.pdf#page=1&selection=88,7,90,34&color=yellow|ngai2011_data-mining-financial-fraud-detection-review, p.1]]
> > of this review clearly show that data mining techniques have been applied most extensively to the detection of insurance fraud, although corporate fraud and credit card fraud have also attracted a great deal of attention in recent years.
>
> Página 559 da revista. Sustenta a atenção recebida pela fraude com cartão de crédito.

^ngai2011-p1-2

> [!PDF|yellow] [[ngai2011_data-mining-financial-fraud-detection-review.pdf#page=1&selection=94,50,95,73&color=yellow|ngai2011_data-mining-financial-fraud-detection-review, p.1]]
> > The main data mining techniques used for FFD are logistic models, neural networks, the Bayesian belief network, and decision trees,
>
> Página 559 da revista. Sustenta as técnicas mais utilizadas.

^ngai2011-p1-3

## Dificuldades da detecção

> [!PDF|yellow] [[dalpozzolo2018_credit-card-fraud-realistic-modeling.pdf#page=1&selection=28,21,33,15&color=yellow|dalpozzolo2018_credit-card-fraud-realistic-modeling, p.1]]
> > this problem involves a number of relevant challenges, namely: concept drift (customers' habits evolve and fraudsters change their strategies over time), class imbalance (genuine transactions far outnumber frauds), and verification latency (only a small set of transactions are timely checked by investigators).
>
> Página 3784 da revista. Sustenta as três dificuldades: mudança de hábitos e de estratégias, desbalanceamento e verificação de poucas transações a tempo.

^dalpozzolo2018-p1

> [!PDF|yellow] [[sulaiman2022_review-machine-learning-credit-card-fraud.pdf#page=3&selection=33,0,37,17&color=yellow|sulaiman2022_review-machine-learning-credit-card-fraud, p.3]]
> > Fraudulent transaction diversity and imbalanced datasets is also a big challenge in CCFD [22]. Getting real-time datasets of credit card transactions is quite challenging. Banks and financial sectors do not expose their customer's data due to GDPR.
>
> Página 57 da revista. Sustenta a restrição de acesso a dados reais. A afirmação sobre a GDPR é do próprio artigo, sem referência.

^sulaiman2022-p3

## Abordagem supervisionada e detecção de anomalias

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=6&selection=45,0,54,20&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.6]]
> > Supervised learning methods treat fraud detection as a binary classification problem and they aim to model the conditional distribution over the two classes fraud and genuine given training data.
>
> Sustenta a detecção supervisionada como classificação binária aprendida com dados rotulados.

^jurgovsky2018-p6

> [!PDF|yellow] [[dalpozzolo2018_credit-card-fraud-realistic-modeling.pdf#page=1&selection=258,26,265,50&color=yellow|dalpozzolo2018_credit-card-fraud-realistic-modeling, p.1]]
> > Alerts are then inspected by professional investigators that contact the cardholders to determine the true nature (either genuine or fraudulent) of each alerted transaction. By doing this, investigators provide a feedback to the system in the form of labeled transactions,
>
> Página 3784 da revista. Com o grifo do sistema típico de Jurgovsky et al. (2018), sustenta que os rótulos dependem da verificação feita pelos investigadores.

^dalpozzolo2018-p1-2

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=7&selection=39,0,43,30&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.7]]
> > Anomaly detection methods take a different perspective. Such methods typically learn a model of normal instances, i.e. genuine transactions, and then identify those transactions that do not conform to the model as anomalous. These methods require the definition of a suitable anomaly score that quantifies the degree of an instance to be abnormal.
>
> Sustenta a detecção de anomalias e a necessidade de um escore de anomalia.

^jurgovsky2018-p7

> [!PDF|yellow] [[ghosh2024_parameter-free-outlier-detection-hdbscan-glosh.pdf#page=2&selection=188,0,189,67&color=yellow|ghosh2024_parameter-free-outlier-detection-hdbscan-glosh, p.2]]
> > Unsupervised outlier detection is based on the premise that potential outliers lie in low-density regions of the reference set.
>
> Sustenta a premissa de que as anomalias estão em regiões de baixa densidade. Mesmo grifo da nota do HDBSCAN\*.

^ghosh2024-p2

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=2&selection=19,0,20,27&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.2]]
> > An integrated framework for density-based cluster analysis, outlier detection, and data visualization is introduced in this article.
>
> Página 1 na numeração da revista. Com o trecho seguinte, sustenta que o HDBSCAN\* produz o agrupamento e a detecção de *outliers* a partir da mesma hierarquia.

^campello2015-p2

> [!PDF|yellow] [[campello2015_hierarchical-density-estimates-hdbscan.pdf#page=2&selection=25,105,30,24&color=yellow|campello2015_hierarchical-density-estimates-hdbscan, p.2]]
> > It can also be further postprocessed so that: (i) a normalized score of "outlierness" can be assigned to each data object, which unifies both the global and local perspectives of outliers into a single definition; and (ii) a "flat" (i.e., nonhierarchical) clustering solution composed of clusters extracted from local cuts through the cluster tree (possibly corresponding to different density thresholds) can be obtained, either in an unsupervised or in a semisupervised way.
>
> Página 1 na numeração da revista. A mesma hierarquia fornece o escore de *outlier* e a partição.

^campello2015-p2-2

## Objetivos específicos

> [!PDF|yellow] [[moulavi2014_density-based-clustering-validation.pdf#page=1&selection=119,0,121,28&color=yellow|moulavi2014_density-based-clustering-validation, p.1]]
> > Because internal validity criteria measure clustering quality based solely on information intrinsic to the data
>
> Sustenta que o DBCV mede a qualidade do agrupamento sem rótulos. Mesmo grifo da nota de Trabalhos Correlatos, e o índice é detalhado na Seção 3.4.8.

^moulavi2014-p1

> [!PDF|yellow] [[davis2006_relationship-precision-recall-roc-curves.pdf#page=1&selection=108,0,112,9&color=yellow|davis2006_relationship-precision-recall-roc-curves, p.1]]
> > Precision-Recall (PR) curves, often used in Information Retrieval (Manning & Schutze, 1999; Raghavan et al., 1989), have been cited as an alternative to ROC curves for tasks with a large skew in the class distribution
>
> Página 233 dos anais. Sustenta a AUPRC como adequada a bases com forte desbalanceamento. Mesmo grifo da nota de métricas de avaliação.

^davis2006-p1

## Hipótese

> [!PDF|yellow] [[jurgovsky2018_sequence-classification-credit-card-fraud.pdf#page=10&selection=16,0,25,82&color=yellow|jurgovsky2018_sequence-classification-credit-card-fraud, p.10]]
> > Either way, in feature space frauds very well intermingle with genuine transactions because of two reasons. First, genuine purchase actions from millions of customers naturally cover a broad spectrum of variability. And secondly, fraudsters apply various hard to detect, yet rational, strategies to execute fraudulent acts that span several customer accounts across varying time periods – but in the end, these acts will likewise only appear as individual transactions in a dataset.
>
> Sustenta o contraponto à hipótese: as fraudes podem se misturar às transações legítimas no espaço dos atributos.

^jurgovsky2018-p10

> [!PDF|yellow] [[dalpozzolo2018_credit-card-fraud-realistic-modeling.pdf#page=1&selection=268,17,281,25&color=yellow|dalpozzolo2018_credit-card-fraud-realistic-modeling, p.1]]
> > The vast majority of transactions cannot be verified by investigators for obvious time and cost constraints. These transactions remain unlabeled until customers discover and report frauds, or until a sufficient amount of time has elapsed such that nondisputed transactions are considered genuine. Thus, in practice, most of supervised samples are provided with a substantial delay,
>
> Página 3784 da revista. Sustenta que os rótulos são escassos ou chegam com atraso.

^dalpozzolo2018-p1-3

