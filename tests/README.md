# Testing Lumis

```bash
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=html
```

Open `htmlcov/index.html` to inspect covered lines and missing branches per module.
CI runs the same suite on Python 3.11–3.13, requires at least 90% combined line and
branch coverage, and retains HTML/XML reports as workflow artifacts. Run one small
section with `uv run --no-sync pytest tests/test_memory.py -q`.

## Sections

Each test file focuses on a small component or related pair of components.

| Area | Test files |
| --- | --- |
| Data, schemas, serialization, tokenizers | `test_core_data.py`, `test_core_helpers_edges.py` |
| Cache, status, scalar and async helpers | `test_cache.py`, `test_status_manager.py`, `test_scalar_utils.py`, `test_async_utils.py` |
| Logging, events, inference helpers | `test_logging_inference.py`, `test_event_emitter.py` |
| Memory and optional Django model | `test_memory.py`, `test_django_memory.py` |
| OpenAI | `test_llm.py`, `test_openai_chat.py`, `test_openai_responses.py` |
| Gemini and Ollama | `test_gemini.py`, `test_ollama_requests.py`, `test_llm.py` |
| Embeddings, FAISS, retrieval | `test_embedding.py`, `test_faiss_storage.py`, `test_vector_similarity.py`, `test_retrieval_engine.py` |
| Agents | `test_base_agents.py`, `test_react_agent.py`, `test_legacy_graph_agent.py` |
| Research sources and tools | `test_research_models.py`, `test_source_tracker.py`, `test_research_tools.py` |
| Graphs and pipelines | `test_graph.py`, `test_pipeline.py`, `test_pipeline_nodes.py`, `test_refinement_pipeline.py`, `test_qa_pipeline.py` |
| NLP and evaluators | `test_semantic_parser.py`, `test_summarizers.py`, `test_fact_extractor.py`, `test_ner.py`, `test_evaluators.py` |
| Search providers | `test_arxiv.py`, `test_wiki.py`, `test_search_engine.py`, `test_market_wrappers.py` |
| Scraping and PDF reader | `test_scraper.py`, `test_pdf_reader.py` |

## Static typing contracts

Run `uv run pyright --warnings` to check the LLM adapters, BaseAgent/ReAct,
LLM pipeline nodes, and the refinement/research pipelines. CI runs this check.
`tests/typecheck` uses `assert_type` to verify that structured completions preserve
their Pydantic model type, nullable parsed results, and scalar/list return shapes.
`test_static_contracts.py` separately checks deliberately invalid calls and requires
the expected diagnostics, so silently degrading a result to `Any` fails the tests.
The package includes `py.typed` so downstream checkers can use these annotations.

`BaseLLM` provides shared bookkeeping and middleware, not a common completion API.
Agents that use OpenAI chat/tool methods explicitly require `OpenAILLM`; this
corrects an overly broad annotation and does not provide provider interchangeability.
Pipelines preserve their provider through a third generic parameter (defaulting to
`BaseLLM` for existing two-parameter annotations). `_require_llm()` checks the
optional provider before an LLM-dependent node uses it.

Middleware callbacks must preserve the input response type for every supported
response, using a generic callable contract. Provider-specific inspection should
narrow the response with `isinstance` and return other response types unchanged.

Cache equality holds both cache locks in object-ID order, with an immediate
self-comparison shortcut. Concurrency tests check reverse comparisons and ensure
writers cannot change either cache during comparison.

## Isolation

- An automatic fixture blocks IPv4/IPv6 connections. Model providers and network
  clients must be injected or patched before calling the component under test.
- Optional search/spaCy modules use scoped fake SDK modules; no credentials or
  pretrained models are needed. Assertions validate Lumis behavior, not SDK internals.
- FAISS tests use real, tiny local indexes with fixed NumPy vectors. Persistence
  and PDF tests use temporary files or in-memory buffers.
- Django is a development-only dependency as well as a production extra. Tests
  instantiate an unsaved concrete model and never access a database.
- Concurrency tests use events/barriers and bounded waits. The running-event-loop
  regression uses a subprocess timeout so a deadlock cannot hang the test runner.

These are unit tests. They do not prove live provider API compatibility, model
quality, installed spaCy model behavior, or production database/service integration.

## Regression fixes found while adding coverage

Tests accompany fixes for shared agent memory, truncated memory limits and deep
copies, logger/event/reset behavior, UTC conversion, large-number formatting,
cache self-comparison deadlocks, and synchronous coroutine execution inside an
active event loop.

Provider regressions include Gemini streaming and grounding indices, OpenAI
reasoning/tool continuation payloads, research-source routing and paper-summary
alignment, search provider mapping/options/awaiting, result-limit cache separation,
Wikipedia language races, and arXiv PDF concurrency.

Other regressions cover vector deletion/MMR/empty retrieval, semantic sentence
boundaries, sentence-length and cosine metrics, scraper failure/header/link handling,
NLP relation extraction, and QA-pipeline serialization and retry limits. Existing
behavior changes are intentional only where a regression test demonstrates the bug.

## Known boundaries

`chroma_vector_db.py` contains only commented-out code and `llm_node.py` is an empty
placeholder. `SemanticParser.parse_documents` explicitly raises `NotImplementedError`;
use its asynchronous batch method. DuckDuckGo search remains an unimplemented stub.
The retrieval engine's deprecated cache helper is not part of its active search path.
Coverage reports expose these gaps rather than hiding them with exclusions.

`BaseAgent.call_tool` currently converts tool exceptions to `None`; tests preserve
that existing contract. Consumers needing explicit error events should account for
this behavior. `run_sync` must not receive awaitables that depend on its blocked
caller's event loop.

FAISS now honors its requested Flat index type, enabling deletion but using an
exact vector scan rather than HNSW approximate search. Deleting from an older
persisted HNSW index rebuilds it once from stored vectors and IDs. Loading legacy
chunks restores missing vectors for MMR without provider calls, with memory cost
proportional to the missing vectors. Wikipedia SDK calls are serialized around
its process-global language setting to prevent concurrent language mix-ups.
