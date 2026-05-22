# arxiv-nlp-pipeline

> **Como avaliar este projeto:** abra o `notebook.ipynb` diretamente no
> GitHub — todas as células estão executadas, com figuras e tabelas
> embutidas. A síntese principal está na seção 5.2.

Pipeline de NLP sobre abstracts científicos do arXiv. Analisa 5.400 artigos de cinco
áreas (cs.LG, stat.ML, q-bio.NC, econ.EM, eess.SP) para caracterizar o vocabulário de
cada disciplina, identificar padrões de publicação, extrair tópicos, treinar
classificadores e construir um grafo de co-ocorrência de conceitos técnicos.

**Narrativa central:** como o ferramental estatístico e de ML circula entre disciplinas
científicas distintas.

---

## Estrutura do repositório

```
arxiv-nlp-pipeline/
├── fetch_corpus.py          # script de download (opcional — ver abaixo)
├── notebook.ipynb           # pipeline completo, fases 1–5
├── data/
│   └── corpus_raw.parquet   # snapshot commitado do corpus
├── outputs/                 # figuras e grafo interativo gerados pelo notebook
├── requirements.txt         # dependências com versões fixadas
└── README.md
```

---

## Corpus

O corpus é derivado do **arXiv Metadata Dataset** disponível no Kaggle:

> <https://www.kaggle.com/datasets/Cornell-University/arxiv>

O arquivo `data/corpus_raw.parquet` contém o snapshot utilizado neste projeto
(5.400 abstracts, selecionados pelo `fetch_corpus.py` com filtro de data mais
recente por categoria). **Não é necessário baixar o dataset para reproduzir o
notebook** — o snapshot já está versionado no repositório.

Se quiser regenerar o corpus a partir do JSON completo do Kaggle:

1. Baixe `arxiv-metadata-oai-snapshot.json` na pasta `data/`.
2. Execute `python fetch_corpus.py` (ou `uv run python fetch_corpus.py`).

**Data do snapshot:** 2026-05-17

---

## Reprodução

### Com uv (recomendado)

```bash
# 1. Instalar dependências
uv sync

# 2. Baixar o modelo spaCy
uv run python -m spacy download en_core_web_sm

# 3. Abrir o notebook
uv run jupyter notebook notebook.ipynb
```

### Com pip

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar o modelo spaCy
python -m spacy download en_core_web_sm

# 4. Abrir o notebook
jupyter notebook notebook.ipynb
```

---

## Ordem de execução

Execute as células do `notebook.ipynb` **em sequência do início ao fim** com um
kernel limpo. O notebook está organizado em cinco fases:

| Fase | Conteúdo |
|------|----------|
| **Fase 1** | Carregamento, limpeza de LaTeX, tokenização, stopwords, stemming vs. lemmatização, POS tagging, análise de vocabulário |
| **Fase 2** | BoW e TF-IDF, n-gramas, Word2Vec exploratório, similaridade por cosseno, motor de busca textual, t-SNE por documento |
| **Fase 3** | Topic modeling (LDA + NMF), classificação supervisionada (ComplementNB, LinearSVC, LogisticRegression), SMOTE, matrizes de confusão |
| **Fase 4** | NER padrão (spaCy), noun chunks + TF-IDF para termos técnicos, EntityRuler customizado, regex, normalização Levenshtein, grafo NetworkX, centralidade, PyVis |
| **Fase 5** | Verificação de reprodutibilidade, síntese para stakeholder |

O notebook não acessa a rede em nenhum momento (`DOWNLOAD = False` na primeira célula).

---

## Reprodutibilidade

Todos os modelos estocásticos usam `random_state=42` ou `seed=42`:

| Componente | Parâmetro |
|---|---|
| Word2Vec | `seed=42`, `workers=1` |
| t-SNE (Word2Vec e TF-IDF) | `random_state=42` |
| TruncatedSVD (pré-t-SNE) | `random_state=42` |
| LDA e NMF | `random_state=42` |
| `train_test_split` | `random_state=42` |
| LinearSVC e LogisticRegression | `random_state=42` |
| SMOTE | `random_state=42` |
| `betweenness_centrality` | `seed=42` |
| `spring_layout` | `seed=42` |

---

## Saídas geradas

| Arquivo | Conteúdo |
|---|---|
| `outputs/fig_abstract_lengths.png` | Distribuição de comprimento de abstracts |
| `outputs/fig_pos_distribution.png` | Distribuição de categorias gramaticais |
| `outputs/fig_wordcloud.png` | Nuvem de palavras (lemas) |
| `outputs/fig_top_terms.png` | Top 25 termos por frequência |
| `outputs/fig_top_terms_by_cat.png` | Top termos por categoria |
| `outputs/fig_ngrams.png` | Top 20 bigramas e trigramas |
| `outputs/fig_w2v_tsne.png` | t-SNE do espaço Word2Vec |
| `outputs/fig_cosine_dist.png` | Distribuição de similaridade por cosseno |
| `outputs/fig_tsne_tfidf.png` | t-SNE dos documentos TF-IDF por categoria |
| `outputs/fig_lda_heatmap.png` | Peso médio dos tópicos LDA por categoria |
| `outputs/fig_nmf_heatmap.png` | Peso médio dos tópicos NMF por categoria |
| `outputs/fig_confusion_base.png` | Matrizes de confusão — baseline |
| `outputs/fig_confusion_lr_comparison.png` | Regressão Logística: balanced vs. SMOTE |
| `outputs/fig_ner_entities.png` | Top entidades NER (ORG, PERSON, GPE) |
| `outputs/fig_knowledge_graph.png` | Grafo de co-ocorrência — visualização estática |
| `outputs/grafo.html` | Grafo interativo PyVis (abrir no browser) |

## Para o avaliador

Este repositório é a entrega completa do trabalho. O notebook está
executado e auto-explicativo. Caso prefira leitura offline, há um PDF
do notebook em `entregavel/notebook.pdf` (no anexo do Moodle).

Resumo dos achados em **5.2 — Cinco áreas, um idioma emergente**, no
fim do notebook.

# Pacotes para submissão
entregavel/