# lumis

An AI agent framework for building LLM-powered applications with multi-provider LLM support, graph-based workflows, vector storage, and NLP pipelines.

## Features

- **Multi-provider LLMs** — Unified interface across OpenAI, Gemini, Ollama, and Perplexity with middleware support and automatic token counting
- **Agents** — ReAct, research, and graph-based agent architectures with event-driven lifecycle hooks
- **Graph Workflows** — Composable, stateful DAG execution with sync and async node support
- **Embeddings** — OpenAI and HuggingFace embedding backends
- **Vector Storage** — Chroma and FAISS vector database integrations
- **NLP** — Named entity recognition, summarization, semantic parsing, and coreference resolution
- **Search & Tools** — Web scraping, arXiv, Wikipedia, Yahoo Finance, Google Trends, and Tavily integrations
- **Memory** — Pluggable memory backends for stateful agent conversations

## Installation

```bash
pip install lumis-ai
```

### Optional extras

Some integrations are opt-in to keep the base install lighter:

| Extra      | What it adds                                  | Install                           |
|------------|-----------------------------------------------|-----------------------------------|
| `spacy`    | NER, fact extraction, coreference resolution  | `pip install lumis-ai[spacy]`     |
| `django`   | Django ORM memory backend                     | `pip install lumis-ai[django]`    |

The `spacy` extra requires language models. After installing, download them:

```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

## License

MIT — see [LICENSE](LICENSE) for details.
