> [!check] Status — em uso na monografia
> - `krstajic2014` — consta do `refbib.bib`, citado em Fundamentação Teórica › Pipeline de Desenvolvimento e Validação › Validação Cruzada Aninhada.
> - `berrar2024`: consta do `refbib.bib`, citado em Fundamentação Teórica › Pipeline de Desenvolvimento e Validação › Validação Cruzada Aninhada, no texto e na fonte das Figuras 3.1 e 3.2.

> [!PDF|yellow] [[krstajic2014_cross-validation-pitfalls.pdf#page=10&selection=3,0,19,50&color=yellow|krstajic2014, p.9]]
> > The nested cross-validation loss estimates differed significantly compared with the cross-validation estimates of the best model on at least caco-QuickProp, Melting Point Mutagen and PLD datasets. This confirms previous reports in the literature (Varma and Simon [9]). Model assessment using repeated nested cross-validation (Figures 10, 11, 12, 13, 14 and 15) showed large variation of loss estimates across the nested cross-validation runs. For example, the proportion misclassified estimate for bbb2 varied between approximately 0.13 and 0.23 (Figure 13). In practical terms, this means that the best model selected on this dataset may have large-sample performance of anywhere between 13% and 23%. Whether this is adequate for a particular application is a domain-dependent question, however we point out that the repeated nested crossvalidation provides the means to make an informed decision regarding the acceptance of the best model.
> 
> *Cross-validation pitfalls when selecting and assessing regression and classification models*
> Krstajic et al.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=5|berrar2024, p.5]]
> > In k-fold cross-validation, the available learning set is partitioned into k disjoint subsets of approximately equal size. Here, "fold" refers to the number of resulting subsets. This partitioning is performed by randomly sampling cases from the learning set without replacement. The model is trained on k-1 subsets, which, together, represent the training set. Then, the model is applied to the remaining subset, which is denoted as the validation set, and the performance is measured. This procedure is repeated until each of the k subsets has served as validation set. The average of the k performance measurements on the k validation sets is the cross-validated performance.
>
> Sustenta a definição da validação cruzada *k-fold*. A Figura 1 do artigo, na mesma página, é a base da Figura 3.1.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=6|berrar2024, p.6]]
> > Cross-validation often involves stratified random sampling, which means that the sampling is performed in such a way that the class proportions in the individual subsets reflect the proportions in the learning set.
>
> Sustenta a frase sobre a amostragem estratificada.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=6|berrar2024, p.6]]
> > Nested cross-validation is a special case of ordinary k-fold cross-validation (CV) that involves one outer CV and several inner CVs [13]. The outer CV is the same as the k-fold cross-validation described in Section 4.3. Each fold of the outer CV involves an internal cross-validation that uses the respective training set. [...] The purpose of the inner CV is to find the best hyperparameters of a learning algorithm.
>
> Sustenta a descrição dos dois níveis da validação cruzada aninhada.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=7|berrar2024, p.7]]
> > Figure 2: (a) 5 × 2 nested cross-validation (CV). The outer CV is a 5-fold cross-validation to evaluate the model fitting procedure. [...] Each fold consists of an inner CV to find the optimal hyperparameter, h. The inner CV is only shown for the fifth fold [...]
>
> Base da Figura 3.2.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=8|berrar2024, p.8]]
> > The optimal value for the fifth inner CV is $h_{opt,5} = 3$, since the average accuracy is the highest. The optimal hyperparameter $h_{opt,k}$ is then used for the model that is to be trained on $R_k$. This model is then applied to $V_k$. [...] By proceeding analogously for all five outer folds, we achieve a cross-validated accuracy of 0.85. The optimal values of the hyperparameters are not necessarily the same for each outer fold [...]
>
> Sustenta a seleção pela melhor média do ciclo interno, o treino com todo o conjunto de treinamento externo e a possibilidade de configurações distintas entre as rodadas.

> [!PDF|yellow] [[berrar2024_cross-validation.pdf#page=11|berrar2024, p.11]]
> > Nested cross-validation disentangles hyperparameter tuning from the model evaluation, but in practice, the less computationally expensive ordinary (or flat) cross-validation is often sufficient despite its optimistic bias.
>
> Sustenta a separação entre a seleção de hiperparâmetros e a avaliação do modelo.

> [!warning] Contraponto ao critério do HDBSCAN* (não citado no texto)
> [[berrar2024_cross-validation.pdf#page=8|berrar2024, p.8]]
> > It would not be advisable to select that value for which the best performance among the five outer folds was achieved because this approach would ignore the results from the remaining four folds and could be just due to chance. Instead, to find the "best" hyperparameter for the "winner" algorithm, it is preferable to run additional ordinary k-fold cross-validations (that is, without nesting), one for each distinct value of $h_{opt,k}$. The hyperparameter for which the best cross-validated performance is achieved will then be selected.
>
> O HDBSCAN* é configurado pelo maior DBCV entre os vencedores dos ciclos internos (Seção 3.1.3.4), critério que este trecho desaconselha. A justificativa da escolha está pendente.



