# Normas da monografia (UFLA e ABNT)

Este arquivo orquestra os documentos de norma e de modelo da pasta `docs/ufla-standards/`. Ele reúne as regras de estrutura e de formatação que valem para a monografia, com a página de onde cada uma saiu, mostra como a classe `uflamon` as implementa e acompanha o que ainda falta ajustar no texto. Antes de mudar a estrutura, a formatação, as citações ou as referências da monografia, consultar este arquivo e, em caso de dúvida, a página indicada do documento original.

Os documentos ficam fora do git (material de terceiros). Só este arquivo é versionado.

## 1. Precedência

1. **Manual de normalização e estrutura de trabalhos acadêmicos da UFLA, 6. ed., 2025.** É a referência principal. Quando a ABNT oferece opções, vale a escolha do manual (destaque em negrito nas referências, *et al.* em itálico depois do primeiro autor, sistema autor-data). Segue a NBR 14724:2024 e a NBR 10520:2023.
2. **Template DOCX 2025-2 da UFLA.** Aplicação do manual em Word. Serve para tirar dúvidas visuais (tamanhos, alinhamentos, espaçamentos). O exemplo é de uma dissertação, então as variantes de TCC de graduação vêm do manual.
3. **Classe `uflamon` e exemplo de Uchôa (2016).** Implementação em LaTeX, feita sobre o manual de 2016 (2. ed.). Continua sendo o modelo de uso da classe. Quando diverge do manual de 2025, vale o manual, e a divergência entra na Seção 5.
4. **NBRs da pasta.** São edições antigas (NBR 14724:2011 e NBR 6023:2002) e servem só de contexto. As normas vigentes que o manual segue são a NBR 6022:2018, a NBR 6023:2018 (revisada em 2020), a NBR 6024:2012, a NBR 6027:2012, a NBR 6028:2021, a NBR 6033:2022, a NBR 6034:2004, a NBR 10520:2023, a NBR 14724:2024, a NBR 17225:2025 e as normas de apresentação tabular do IBGE (1993) (manual, p. 21).

Os comentários da orientação prevalecem nas escolhas de conteúdo e nas que o manual deixa em aberto.

## 2. Documentos da pasta

| Arquivo | Nome original | O que é | Situação | Leitura |
|---|---|---|---|---|
| `ufla-standardization-manual-6th-ed-2025.pdf` | `LIVRO_Manual de normalização e estrutura de trabalhos acadêmicos. 6. ed 2025-2.pdf` | Manual da UFLA, 6. ed. (2025), 150 p. | Vigente, referência principal | Lido inteiro. Das Seções 5.4.7 a 5.4.10.11 (patentes, documentos jurídicos, audiovisuais, iconográficos e cartográficos) só os títulos, por não se aplicarem |
| `ufla-tcc-template-2025-2.docx` | `TEMPLATE_tcc_grad__2025-2_DOCX.docx` | Template Word do padrão UFLA (exemplo de dissertação, 2024) | Vigente | Lido inteiro, com estilos e medidas de página |
| `ufla-approval-page-example.docx` | `exemplo-folhaaprovação.docx` | Folha de aprovação de monografia de Ciência da Computação | Modelo antigo (2015 e 2016, papel carta, margens de 4,5 cm) | Lido inteiro |
| `uflamon-template/` | `Uflamon-Template/` | Classe `uflamon` e exemplo de Uchôa (2016), 3. ed. | Implementação em LaTeX do manual de 2016 | `exemplo_monografia_UFLA.tex`, o PDF compilado, `elementos.tex` e `uflamon.cls` lidos. O `referencias.tex` é uma cópia em LaTeX do capítulo de referências do manual antigo |
| `abnt-nbr-14724-2011.pdf` | `ABNT_2011.pdf` | NBR 14724:2011, trabalhos acadêmicos | Substituída pela NBR 14724:2024 (comparação no Apêndice A do manual, p. 146-147) | Lida inteira |
| `abnt-nbr-6023-2002.pdf` | `abntnbr6023.pdf` | NBR 6023:2002, referências | Substituída pela NBR 6023:2018, revisada em 2020 | Seções 1 a 7.7 lidas, as demais só pelos títulos |

A `uflamon.cls` da pasta `uflamon-template/` é mais antiga que a de `docs/monografia/` (papel carta, fonte 11 e margens de 4,5 cm, contra A4, fonte 12 e margens de 3 cm e 2 cm na revisão de 09/08/2016). A monografia usa a de `docs/monografia/`.

## 3. Regras

### 3.1 Estrutura (manual, p. 33-34 e Figura 3.1)

| Elemento | No TCC de graduação | Observação |
|---|---|---|
| Capa | Obrigatório | |
| Lombada | Opcional | Só em versão impressa, pela NBR 12225 |
| Folha de rosto | Obrigatório | |
| Ficha catalográfica | Não obrigatória | Exigida só na pós-graduação *stricto sensu* (p. 42) |
| Errata | Opcional | |
| Folha de aprovação | Obrigatório | |
| Dedicatória, agradecimentos, epígrafe | Opcionais | |
| Resumo e resumo em língua estrangeira | Obrigatórios | |
| Indicadores de impacto | Não exigidos | Exigidos só na pós-graduação *stricto sensu* (p. 51) |
| Listas de ilustrações, tabelas, siglas e símbolos | Opcionais | |
| Sumário | Obrigatório | Último elemento pré-textual |
| Introdução, desenvolvimento e conclusão | Obrigatórios | |
| Referências | Obrigatório | |
| Glossário, apêndices, anexos, índice | Opcionais | |

Os elementos pré-textuais ficam em português. O título é idêntico ao registrado na ata de defesa (p. 35, 38 e 44).

### 3.2 Capa (p. 35-37)

Todos os elementos centralizados e distribuídos uniformemente, nesta ordem: a logomarca mais recente da UFLA; o nome do autor em maiúsculas, fonte Times ou similar, tamanho 14, negrito, espaço simples; o título em maiúsculas, tamanho 16, negrito, espaço 1,5, com o subtítulo depois de dois pontos e sem negrito (o modelo da Figura 3.3 o mostra em maiúsculas); a cidade, hífen e a sigla do estado em maiúsculas, tamanho 14, negrito; o ano de depósito, tamanho 14, negrito, no fim da página. Fontes similares aceitas incluem Computer Modern Serif, Liberation Serif e Nimbus Roman (nota 2 da p. 35).

### 3.3 Folha de rosto (p. 38-40)

Fonte 12, espaço simples, exceto no título. O autor vem no alto, centralizado, em maiúsculas e negrito. O título vem em maiúsculas e negrito, com o subtítulo depois de dois pontos, sem negrito e com espaço 1,5. A nota descritiva vai do meio da página à margem direita, em espaço simples. O orientador e o coorientador são precedidos da titulação abreviada e centralizados (o modelo da Figura 3.4 traz "Prof.(a) Dr.(a) Xxxxx" e, abaixo, "Orientador(a)"). Fecham a página a cidade e a sigla do estado, em maiúsculas e negrito, e o ano de depósito em negrito.

Nota descritiva do TCC de bacharelado (Quadro 3.1, p. 39): "Trabalho de conclusão de curso apresentado à Universidade Federal de Lavras, como parte das exigências do Curso de [nome do curso], para a obtenção do título de Bacharel." O exemplo de folha de aprovação da pasta usa "Curso de Ciência da Computação".

### 3.4 Folha de aprovação (p. 44-46)

Fonte 12, espaço simples, nesta ordem: o nome do autor; o título em português; o título em inglês, com a mesma formatação; a nota descritiva; "APROVADA em" com a data da ata, à esquerda; os examinadores com titulação e instituição, à esquerda; o orientador e o coorientador com a titulação abreviada, centralizados, com espaço para a assinatura do orientador; a cidade e a sigla do estado; o ano de depósito.

### 3.5 Titulação dos orientadores (p. 38 e 45, Figuras 3.4 e 3.8)

A titulação abreviada precede o nome. O manual usa "Prof.(a) Dr.(a)", o template usa "Prof. DSc." e "Prof.ª DSc." e o exemplo da `uflamon` usa "Prof. DSc.", "Prof. MSc." e "Profa. Esp.". Na monografia, a orientadora é a Profa. Dra. Elaine Cecilia Gatto e o coorientador é o Prof. Me. Natanael Fabrício Dacioli Batista. Com `\orientadora`, a classe escreve "Orientadora" abaixo do nome.

### 3.6 Dedicatória, agradecimentos e epígrafe (p. 46-49)

A dedicatória não tem título e fica na parte inferior da página, do meio da mancha gráfica à margem direita. Os agradecimentos têm título em maiúsculas, negrito e centralizado. A epígrafe traz uma citação com a indicação de autoria, na parte inferior, do meio da mancha à direita, e também pode abrir as seções primárias.

### 3.7 Resumo e abstract (p. 49-51)

Página própria, parágrafo único, sem enumeração de tópicos, verbo na terceira pessoa e de 150 a 500 palavras (NBR 6028:2021). O título RESUMO vem centralizado, em maiúsculas e negrito, e o texto é justificado, em espaço simples. Abaixo do texto vem "Palavras-chave:" com os termos separados por ponto e vírgula e terminados em ponto, com inicial minúscula, exceto nomes próprios e científicos. O abstract segue a mesma forma, com "Keywords:".

O texto de exemplo do template ainda descreve a regra antiga (termos separados por ponto, com inicial maiúscula), mas a linha de palavras-chave do próprio template já usa ponto e vírgula. O exemplo da `uflamon` segue a regra antiga. Vale o manual.

### 3.8 Listas e sumário (p. 54-60)

Cada item das listas traz o nome específico, o número, travessão, o título e a página, e o manual recomenda lista própria para cada tipo de ilustração. As siglas vêm em ordem alfabética, e os símbolos, na ordem do texto. No sumário, o título SUMÁRIO vem centralizado, em maiúsculas, negrito e tamanho 12. Os títulos se alinham à esquerda pela margem do indicativo mais extenso, com a mesma grafia do texto. Os elementos pré-textuais não entram no sumário, e os pós-textuais entram sem número.

### 3.9 Formatação geral (p. 68-69)

Papel A4, cor preta, fonte Times ou similar, tamanho 12 no texto e 11 nas citações com mais de três linhas, nas notas, na paginação, na ficha catalográfica e nas fontes e legendas das ilustrações e tabelas. Margens superior e esquerda de 3 cm, inferior e direita de 2 cm, e cabeçalho de 2 cm. Espaço 1,5 em todo o texto, inclusive nos títulos, e simples nas citações longas, notas, referências, legendas, nota descritiva e ficha. Parágrafo com recuo de 1,25 cm, justificado e sem espaço entre parágrafos. Cada seção primária começa em página nova, no alto da mancha gráfica, separada do texto por um espaço 1,5, e as subseções são separadas do texto anterior e do seguinte por um espaço 1,5. Os títulos com mais de uma linha se alinham, a partir da segunda, sob a primeira letra da primeira palavra.

### 3.10 Seções e numeração progressiva (p. 69-73)

Pela NBR 14724:2024, o trabalho não se divide em capítulos, só em seções (p. 69, Figura 3.26, e Apêndice A, p. 146). No texto, as divisões são chamadas de seção ("na Seção 2", "esta seção"), como o próprio manual faz, e não de capítulo.

A numeração usa algarismos arábicos e vai até a seção quinária. O indicativo fica alinhado à esquerda e não é separado do título por ponto nem por outro sinal, e toda seção tem texto próprio. Os títulos sem indicativo (errata, agradecimentos, listas, resumo, abstract, sumário, referências, glossário, apêndices, anexos e índices) são centralizados, em maiúsculas e negrito (p. 72). A capa, a folha de rosto, a folha de aprovação, a dedicatória e a epígrafe não têm título nem número (p. 73).

A introdução é a seção 1. Nela não se recomendam tabelas nem ilustrações, e as citações ficam restritas a definições e relatos históricos (p. 60).

### 3.11 Alíneas e subalíneas (p. 71-72)

O texto que antecede as alíneas termina em dois pontos. As alíneas são indicadas por letra minúscula seguida de parêntese, com recuo, e o texto de cada uma começa em minúscula e termina em ponto e vírgula, exceto o da última, que termina em ponto. As subalíneas começam por travessão seguido de espaço, com recuo em relação à alínea.

### 3.12 Paginação (p. 73)

As páginas são contadas a partir da folha de rosto e numeradas a partir da introdução, em algarismos arábicos, no canto superior direito, em tamanho 11. A ficha catalográfica não é contada. Apêndices e anexos continuam a numeração do texto.

### 3.13 Equações e fórmulas (p. 73-74)

Destacadas no texto e numeradas com algarismos arábicos entre parênteses, alinhados à direita. O modelo do manual numera por seção, como em (4.1).

### 3.14 Ilustrações (p. 74-78)

O título fica acima, no formato "Figura 3.1 – Título", com só a inicial do tipo em maiúscula, o número de ordem (sequencial ou por seção, ambos aceitos, p. 75), travessão e o título, em tamanho 12 e espaço simples. O título longo se alinha sob a primeira letra da primeira palavra e é justificado. Título, ilustração e fonte ficam centralizados.

A legenda, texto que explica a ilustração, fica abaixo, em tamanho 11 e espaço simples. A fonte também fica abaixo, em tamanho 11 e espaço simples, e é obrigatória mesmo quando a ilustração é do autor ("elaboração própria" ou "elaborado pelo próprio autor" no manual, "Fonte: Do autor (2024)." no template).

A ilustração é citada no texto e inserida o mais perto possível do trecho a que se refere, com chamada no texto ("A Figura 3.2 ilustra") ou entre parênteses no fim da frase ("(Figura 3.2)"), só com a inicial maiúscula. Quando ocupa mais de uma página, o título se repete, com "(continua)", "(continuação)" e "(conclusão)".

### 3.15 Tabelas (p. 78-80)

O título segue o mesmo formato das ilustrações (palavra Tabela, número, travessão e título em tamanho 12 e espaço simples). Notas e fonte ficam abaixo, em tamanho 11 e espaço simples, e a fonte é obrigatória. A tabela segue o IBGE (1993). Não é fechada nas laterais e não tem traços horizontais entre os dados. Traços horizontais limitam o topo e o rodapé e separam o cabeçalho, e traços verticais separam só a coluna indicadora, evitados no corpo. Uma linha de total recebe destaque tipográfico, e as frações são escritas em decimais. Uma tabela de apêndice é chamada como "Tabela A.1, do Apêndice A".

A tabela tem o dado numérico como informação central. Conteúdo predominantemente textual vai em quadro, que é uma ilustração.

### 3.16 Citações (p. 81-89, NBR 10520:2023)

Na UFLA, as citações seguem o sistema autor-data (p. 85). Dentro dos parênteses, a autoria vem em maiúsculas e minúsculas, como em (Derrida, 1967, p. 293), (Gong; Tian, 2020) e (Maia; Porte; Souza, 2023). A caixa alta, como em (DERRIDA, 1967), era a regra da NBR 10520:2002.

Com quatro ou mais autores, cita-se o primeiro seguido de *et al.*, de modo uniforme em todo o texto. Vários documentos citados juntos seguem a ordem alfabética, separados por ponto e vírgula dentro dos parênteses, e documentos do mesmo autor e ano recebem letras (2020a, 2020b). Pessoa jurídica é citada pelo nome ou pela sigla, em maiúsculas e minúsculas, com a sigla em maiúsculas (IBGE).

A citação direta de até três linhas fica entre aspas duplas, com a página. A de mais de três linhas tem recuo de 4 cm, tamanho 11, espaço simples e não leva aspas. Os acréscimos à citação são indicados por "tradução nossa", "grifo nosso", [...] para supressões e [ ] para interpolações. Na citação de citação, usa-se apud, e só a fonte consultada entra nas referências. O ponto final encerra a frase, não a citação.

As notas são chamadas só de notas (a NBR 10520:2023 abandonou "nota de rodapé"), com numeração arábica consecutiva, e se dividem em notas de referência e notas explicativas.

### 3.17 Referências (p. 90-139, NBR 6023:2018)

O título REFERÊNCIAS vem em maiúsculas, negrito e centralizado. A lista é única, em ordem alfabética (NBR 6033:2022), alinhada à esquerda, em espaço simples, com as referências separadas por uma linha em branco de espaço simples. O título da obra é destacado em negrito, exceto quando a entrada é pelo próprio título.

A entrada é pelo último sobrenome, em maiúsculas, seguido dos prenomes, abreviados ou não, e os autores são separados por ponto e vírgula. Até três autores, todos aparecem. Com quatro ou mais, o primeiro seguido de *et al.*, em itálico (p. 20 e 93). Elementos complementares, quando usados, entram em todas as referências do mesmo tipo.

Documentos online terminam com "Disponível em:" e "Acesso em:", e o DOI entra quando houver. A NBR 6023:2002 punha o endereço entre os sinais < >, o que a edição de 2018 não usa. Nos eventos, o título do documento é "Anais [...]" ou "Proceedings [...]" (a edição de 2002 usava "Anais...").

Modelos mais usados neste trabalho:

- Artigo: AUTOR. Título do artigo. **Título do periódico**, Local, v. x, n. y, p. inicial-final, data. DOI. Disponível em: URL. Acesso em: data (p. 121-122).
- Trabalho em evento: AUTOR. Título. In: NOME DO EVENTO, número, ano, Local. **Anais [...]**. Local: Editora, ano. p. inicial-final (p. 125-126).
- Livro: AUTOR. **Título**: subtítulo. Edição. Local: Editora, ano (p. 114).
- Parte de livro: AUTOR. Título da parte. In: AUTOR (org.). **Título**. Local: Editora, ano. p. inicial-final (p. 116).
- Trabalho acadêmico: AUTOR. **Título**. Ano. Tipo (Grau em Curso) – Instituição, Local, ano (p. 117).
- Conjunto de dados: AUTOR. **Título** [Conjunto de dados]. Local: Instituição, data. Disponível em: URL. Acesso em: data (p. 138-139).
- Software e base de dados (documento de acesso exclusivo em meio eletrônico): AUTOR. **Título**. Versão. Local: Editora, data. Descrição física do meio eletrônico (p. 137).

Os meses são abreviados no idioma da publicação (Anexo B, p. 150): jan., fev., mar., abr., maio, jun., jul., ago., set., out., nov. e dez. em português, e Jan., Feb., Mar., Apr., May, June, July, Aug., Sept., Oct., Nov. e Dec. em inglês. A edição também segue o idioma do documento (2nd ed., 3rd ed.).

### 3.18 Apêndices e anexos (p. 64-66)

O apêndice é do autor, e o anexo, de terceiros. O título segue o destaque das seções primárias (maiúsculas e negrito), no formato "APÊNDICE A – TÍTULO", com letras dobradas quando o alfabeto se esgota. No texto, aparecem como "Apêndice A" ou "(Apêndice A)", e as páginas continuam a numeração do texto.

### 3.19 Siglas

Na primeira menção, o nome completo vem seguido da sigla entre parênteses (manual, p. 56, e NBR 14724:2011, 5.6).

## 4. Mapa na classe `uflamon` (`docs/monografia/uflamon.cls`)

| Elemento | Comando | Comportamento da classe | Situação |
|---|---|---|---|
| Capa | `\author`, `\title`, `\subtitle`, `\local`, `\date` | Autor em `\Large` (14,4 pt) e negrito, título em `\LARGE` (redefinido para 16 pt) e negrito, subtítulo em maiúsculas sem negrito, cidade e ano em negrito | Conforme |
| Folha de rosto | `\tipo`, `\orientador` ou `\orientadora`, `\coorientador` ou `\coorientadora` | Nota descritiva em caixa de 8 cm à direita, em espaço 1,5, nome na linha de cima e o papel ("Orientadora", "Coorientador") abaixo, sem negrito | Ver pendência 17 (o modelo do manual mostra os nomes em negrito, que o texto do manual não exige) |
| Ficha catalográfica | `\fcautor` e afins | Só aparece se `\fcautor` estiver preenchido | Não usada (não obrigatória na graduação) |
| Folha de aprovação | `\engtitle`, `\engsubtitle`, `\bancaum` a `\bancaquatro{nome}{instituição}`, `\defesa` | Só aparece se `\defesa` não estiver vazio. Escreve "Co-Orientador" ou "Co-Orientadora" | Fora do documento até a defesa. Os dados ficam em `docs/monografia/approval-page.tex`, que volta ao preâmbulo com `\input{approval-page}`. Ver pendências 6 a 8 |
| Resumo e abstract | `\palchaves`, `\resumo`, `\keywords`, `\abstract` | Espaço simples, título em `\large` | Ver pendência 14 |
| Sumário | `\tableofcontents` | SUMÁRIO em negrito, tamanho normal | Conforme |
| Seções | `\chapter` (seção primária), `\section` e abaixo | Primária em maiúsculas e negrito, em página nova. As demais em negrito | Conforme |
| Figuras, tabelas e quadros | ambientes `figure`, `table` e `quadro` | Numeração por seção (3.1), título acima com travessão e alinhamento pendurado, título em `font=small` (cerca de 11 pt) | Mantido (pendência 5) |
| Citação longa | ambiente `quote` | Recuo de 4 cm, `\small`, espaço simples | Conforme |
| Notas | `\footnote` | `footmisc` com alinhamento pendurado, `\small` | Conforme |
| Apêndices | `\apendice{Título}` | "APÊNDICE A -- Título" em negrito, com o título na grafia digitada. No sumário, "APENDICE" sem acento | Ver pendência 9 |
| Referências | `\referencias` e `abntex2cite` com `alf`, `abnt-etal-cite=3`, `abnt-etal-list=3` e `abnt-emphasize=bf` | Título em negrito e *et al.* a partir de quatro autores. Chamadas entre parênteses em caixa alta | Caixa alta mantida (pendência 3). Ver pendência 10 |
| Paginação | estilo `ufla` | Número no canto superior direito, tamanho normal (12) | Ver pendência 13 |

## 5. Pendências da monografia

Cada item traz a regra, a fonte, onde está no projeto e a situação. As mudanças no texto seguem o fluxo de sempre (bloco no `scratch.tex`, aprovação, `main.tex` e commit). Ao resolver um item, marcar `[x]` e anotar o commit.

- [x] **1. Titulação da orientadora e do coorientador e rótulo "Orientadora".** Manual, p. 38 e 45. `main.tex:82-83`. Resolvida no commit `docs: add the UFLA standards guide and the advisors' titles`.
- [x] **2. "Capítulo" vira "seção" no texto.** NBR 14724:2024 (manual, p. 69 e 146). As seções primárias são chamadas pela referência numerada ("Seção 1", "Seção 5"), com rótulos `cap:` em todas elas. Resolvida no commit `docs: call the divisions sections and apply the v4 form fixes`.
- [x] **3. Chamadas entre parênteses em maiúsculas e minúsculas.** NBR 10520:2023 (manual, p. 84-87). O `abntex2cite` gera caixa alta, como em "(HUBERT; DEBRUYNE, 2010)", em todo o texto. Decidido manter a caixa alta, que é o padrão da classe `uflamon` distribuída pela UFLA.
- [x] **4. Palavras-chave e keywords separadas por ponto e vírgula, com inicial minúscula e ponto final.** Manual, p. 50-51. `\palchaves`, `\keywords` e o `pdfkeywords` do `hyperref`. Resolvida no commit `docs: call the divisions sections and apply the v4 form fixes`.
- [x] **5. Título das ilustrações, tabelas e quadros em tamanho 12.** Manual, p. 74 e 78, e template. `uflamon.cls:981-985` (`font=small`). Decidido manter o `font=small`, o mesmo da `uflamon` do exemplo de Uchôa (2016).
- [ ] **6. Título em inglês na folha de aprovação.** Manual, p. 44. `approval-page.tex` (`\engtitle` e `\engsubtitle` vazios).
- [ ] **7. Data de aprovação e banca com titulação e instituição.** Manual, p. 45. `approval-page.tex`. Depende da defesa.
- [ ] **8. Rótulo "Co-Orientador" na folha de aprovação.** A grafia atual é "Coorientador", como na folha de rosto e no manual. `uflamon.cls:758` e `765`.
- [ ] **9. Título do apêndice com o destaque das seções primárias (maiúsculas) e acento em "APÊNDICE" no sumário.** Manual, p. 64-65. `uflamon.cls:1087-1088`.
- [ ] **10. *Et al.* em itálico.** Manual, p. 20. `main.tex:62`. Conferir no PDF. O exemplo da `uflamon` traz a opção `abnt-etal-text=emph`, comentada.
- [ ] **11. Nota descritiva com "Curso de Ciência da Computação".** Manual, Quadro 3.1 (p. 39), e exemplo de folha de aprovação. `main.tex:81` diz "curso de Ciências da Computação". Confirmar o nome oficial do curso.
- [ ] **12. Referência do conjunto de dados no modelo "[Conjunto de dados]".** Manual, p. 138-139. `refbib.bib:146` (`kaggle2018`, `@misc`).
- [ ] **13. Paginação em tamanho 11.** Manual, p. 68. `uflamon.cls:311`. Ajuste menor.
- [ ] **14. Títulos RESUMO e ABSTRACT em `\large`.** O template usa tamanho 12. `uflamon.cls:841` e `866`. Ajuste menor, a conferir.
- [ ] **15. Logomarca mais recente da UFLA na capa.** Manual, p. 35. `docs/monografia/logoufla.png`. Conferir.
- [ ] **16. Autoria da epígrafe.** "(Pierre Bourdie)" em `main.tex:100-102`. Conferir a grafia (Bourdieu) e a indicação da obra.
- [ ] **17. Nota descritiva da folha de rosto em espaço simples.** Manual, p. 38. A classe a imprime com o espaço 1,5 do texto, enquanto a folha de aprovação já usa espaço simples. `uflamon.cls:535`.

Não se aplicam ao TCC de graduação a ficha catalográfica e os indicadores de impacto. As listas de ilustrações, tabelas e siglas são opcionais.

## 6. Manutenção

Ao receber uma nova versão de norma ou modelo, colocá-la nesta pasta com nome em inglês, registrá-la na Seção 2 e revisar as regras afetadas com a página. Ao resolver uma pendência, marcá-la na Seção 5 com o commit correspondente.
