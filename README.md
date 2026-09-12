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
| `search`   | arXiv, Google Trends, Yahoo Finance           | `pip install lumis-ai[search]`    |
| `django`   | Django ORM memory backend                     | `pip install lumis-ai[django]`    |

The `spacy` extra requires language models. After installing, download them:

```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

## Development and checks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run --no-sync pytest -q
uv build
uvx twine check --strict dist/*
```

CI runs the tests on Python 3.11, 3.12, and 3.13 for pull requests and pushes to
`main`. Provider calls in the tests are mocked; no API keys are required.

## Publishing a release

The PyPI distribution is **lumis-ai** (the Python import is `lumis`). Publishing
is triggered by pushing a version tag such as `v0.1.1a8`, not by an ordinary
commit or push to `main`.

### One-time PyPI setup

In [the project's Publishing settings](https://pypi.org/manage/project/lumis-ai/settings/publishing/),
add or verify a GitHub Trusted Publisher with these values:

| Field | Value |
|-------|-------|
| Owner | `tareksanger` |
| Repository | `lumis` |
| Workflow filename | `publish.yml` |
| Environment | Leave blank; this workflow does not set one |

Trusted Publishing uses GitHub's identity token, so no PyPI token secret is
needed. The values must match the workflow; see the
[PyPI setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

### Each release

1. Choose an unused version and update `project.version` in `pyproject.toml`.
   Run `uv lock` to update the local package version in `uv.lock`.
2. Run the development checks above. Commit the intended release changes and
   push them to `main`. Wait for CI to pass.
3. Tag that tested commit with `v` followed by the exact package version. For
   the prepared `0.1.1a8` release:

   ```bash
   git tag -a v0.1.1a8 -m "Release 0.1.1a8"
   git push origin v0.1.1a8
   ```

4. Watch [Publish to PyPI in GitHub Actions](https://github.com/tareksanger/lumis/actions/workflows/publish.yml).
   It runs tests, checks the tag against the package version, builds and validates
   the wheel and source archive, then publishes them.
5. Verify the release on [PyPI](https://pypi.org/project/lumis-ai/) and install it:

   ```bash
   pip install --upgrade "lumis-ai==0.1.1a8"
   ```

Alpha versions require an explicit version or `pip install --upgrade --pre lumis-ai`.
Use a new version for every upload; PyPI does not allow replacing release files.
If no workflow run appears, check that the tag was pushed. If publishing reports
`invalid-publisher`, check the Trusted Publisher fields against the table above.

## License

MIT — see [LICENSE](LICENSE) for details.
