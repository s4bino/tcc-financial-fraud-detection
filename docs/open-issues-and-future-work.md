# Pendências e trabalhos futuros

Registro dos pontos em aberto identificados ao longo do desenvolvimento, com o
motivo pelo qual cada um importa e o que seria preciso fazer. A ordem dentro de
cada seção vai do mais crítico ao menos.

---

## Metodologia

### 1. Critério de seleção: máximo vs. média do DBCV

O ciclo interno elege, em cada fold, a combinação de maior DBCV; a avaliação
externa usa então o **máximo** desses cinco vencedores. O problema é que o
máximo de uma distribuição instável seleciona também o fold mais favorável.

Os resultados atuais mostram a instabilidade com clareza. Para os **mesmos**
parâmetros `min_cluster_size=5, min_samples=300`, ao longo dos cinco folds
internos:

| | mínimo | máximo | amplitude |
|---|---|---|---|
| DBCV | 0,1066 | 0,9590 | 0,852 |
| Pontos de ruído | 7.985 | 130.231 | 122.246 |

Os melhores por fold se dividem em dois regimes:

| Fold interno | `min_cluster_size` | `min_samples` | DBCV | F1 |
|---|---|---|---|---|
| 1 | 75 | 335 | 0,1914 | 0,0109 |
| 2 | 5 | 300 | 0,9574 | 0,0675 |
| 3 | 75 | 335 | 0,1927 | 0,0109 |
| 4 | 5 | 300 | 0,9590 | 0,0698 |
| 5 | 5 | 325 | 0,1922 | 0,0112 |

O critério atual escolhe `(5, 300)` por causa do fold 4 — mas essa mesma
combinação obteve 0,1066 em outro fold. Como os folds são quase idênticos entre
si (diferem em 20% dos dados), uma variação dessa magnitude aponta para
sensibilidade do algoritmo, não para uma configuração genuinamente superior.

**Alternativa a avaliar:** eleger a combinação de melhor **DBCV médio** entre os
folds internos, em vez do maior valor isolado. É o que a validação cruzada
aninhada normalmente faz — a média é a estimativa, o máximo é ruído. Vale
reportar as duas seleções e comparar, já que a divergência entre elas é em si um
resultado sobre a estabilidade do HDBSCAN nesta base.

### 2. Instabilidade do HDBSCAN entre folds

Ampliando o item anterior: no conjunto completo do grid, o DBCV varia de 0,0823
a 0,9590 e o número de pontos classificados como ruído vai de 7.985 a 131.055.
Como fraude é justamente o rótulo de ruído, essa amplitude determina diretamente
quantas transações o modelo acusa.

É a maior ameaça à validade do capítulo de resultados. Precisa ser investigada e
reportada explicitamente, não omitida. Caminhos possíveis: verificar se o regime
de DBCV alto corresponde a uma solução degenerada (poucos clusters gigantes),
repetir os folds com outra semente para separar efeito de partição de efeito de
algoritmo, e reportar desvio padrão junto das médias.

### 3. Referência do DBCV ausente

O DBCV é o critério de seleção de toda a etapa não supervisionada, mas
`moulavi2014` — o artigo que o propõe — não está em `docs/monografia/refbib.bib`
nem é citado na seção de Metodologia. A métrica que governa as escolhas do
trabalho precisa estar fundamentada.

---

## Código

### 4. `run_hdbscan_grid_search.py` não é importável

O arquivo começa com duas linhas `!pip install`, sintaxe de notebook. Isso
impede que ele seja importado, e portanto que seja coberto por testes
automatizados — é o único módulo do projeto fora da suíte do pytest. Mover essas
duas linhas para a documentação (ou para a imagem, no caso da Modal) resolveria.

### 5. `precision_score` sem `zero_division` em `grid_search.py`

`src/models/grid_search.py` chama `precision_score` sem o argumento
`zero_division`, e a execução emite 22 avisos por divisão por zero — situação
frequente quando um modelo não acusa nenhuma fraude. O módulo do HDBSCAN já usa
`zero_division=0`; o supervisionado deveria ser consistente.

### 6. `manual_grid_search` falha com diretório vazio

Se o diretório de folds estiver vazio, a função levanta `NameError` em vez de
uma mensagem útil. O comportamento está documentado por um teste de
caracterização em `tests/models/test_manual_grid_search.py`.

### 7. Divergência entre a grade e os resultados versionados

O bloco `__main__` define uma grade de 3 × 7 = 21 combinações, mas os CSVs em
`results/unsupervised/` têm 25 por fold — foram gerados com uma grade anterior
(5 × 5). Antes de reportar qualquer número na monografia, a grade do código e a
dos resultados precisam coincidir.

---

## Reprodutibilidade

### 8. Baseline supervisionado não versionado

`results/supervised/` está vazio. Sem os resultados do Random Forest e do
XGBoost não há comparação a fazer, que é o objeto do trabalho.
`src/models/run_supervised_grid_search.py` precisa ser executado e sua saída
versionada.

### 9. Sincronizar as duas versões do pipeline não supervisionado

`src/models/run_hdbscan_grid_search_modal.py` replica a lógica do original para
executar na Modal. Toda alteração de lógica no original precisa ser espelhada
nele, ou as duas versões divergem silenciosamente.

---

## Texto

### 10. Campos em branco na monografia

`docs/monografia/main.tex` ainda tem espaços reservados que aparecem no PDF
compilado: `\resumo{ - }`, `\abstract{ - }`, `\bancaum{ - }{UFLA}` e
`\defesa{ - }`. Os capítulos de desenvolvimento também estão vazios.
