# Group 2 Graph Analytics Orchestrator

A Gemini-powered LangGraph orchestrator that plans, executes and chains manifest-discovered graph tools while grounding every answer in traced tool evidence.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key (OpenAI-compatible function calling)
- The published `Registry.zip`, to build the curated registry (see below)

The API key is read from the environment variable **`LLM_API_KEY`** (directly, or from a `.env` file). The provider and model are set with `LLM_BASE_URL` and `LLM_MODEL`. Never commit an API key.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

```env
LLM_API_KEY=your-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.8-flash
```

Instead of a `.env` file you can export the variable: `export LLM_API_KEY=your-key`.

## Curate Registry.zip

The runtime loads exactly one implementation per tool from `registry/tools/`, which you build once from `Registry.zip`. Extract `Registry.zip` anywhere and point `--source` at its `tools/` folder. Curation runs every implementation through the course validator, a canonical workload, a NetworkX-based check and a determinism check, and selects the fastest eligible one per tool:

```bash
python curate_registry.py evaluate \
  --source <path to extracted Registry.zip>/tools \
  --report .orchestrator/curation_report.json \
  --selection .orchestrator/registry_selection.json
```

Review or override `.orchestrator/registry_selection.json`, then create the registry and check it:

```bash
python curate_registry.py materialize \
  --source <path to extracted Registry.zip>/tools \
  --selection .orchestrator/registry_selection.json \
  --dest registry/tools \
  --replace

python curate_registry.py verify
```

Registry source files are copied unchanged. Strict verification requires one implementation for all 33 analytical tools plus the four instructor infrastructure tools; `run_queries.py` refuses to start until `registry/tools/` passes it. Because selection is by speed, a run on another machine may pick a different, equally eligible implementation.

## Run

The required command remains:

```bash
python run_queries.py --queries queries_public.json --out answers_public.json
```

Traces are written to `traces/<query_id>.jsonl`. Checkpoints and Gemini responses are stored in the ignored `.orchestrator/` directory.

Useful development options:

```bash
python run_queries.py \
  --queries queries_public.json \
  --out answers_public.json \
  --resume
```

- `--resume`: skip terminal answers and resume interrupted workflows
- `--retry-errors`: rerun prior error results when resuming
- `--no-cache`: disable Gemini response caching
- `--allow-incomplete-registry`: warn and continue instead of failing on missing/duplicate tools
- `--state-dir`: choose a different checkpoint/cache directory

Strict registry startup is the default. Warn mode is intended only for development.

## Query formats

The runner accepts plain text, JSON arrays, JSON objects containing a `queries` list, and JSONL. Objects may use `query_id`/`id` and `question`/`query`/`text`. `queries_public.json` is the official public set of 30 questions.

## Answer format

Every entry contains exactly the PDF fields:

```json
{
  "query_id": "q07",
  "status": "answered",
  "answer_text": "Alice is the most influential member.",
  "answer": {"type": "node", "value": "n42"},
  "tools_used": ["pagerank.29a7a55d"],
  "trace_file": "traces/q07.jsonl"
}
```

Statuses are `answered`, `clarification_needed`, `unsupported` and `error`. Answer types are `number`, `boolean`, `node`, `node_list`, `path` and `text`.

## Architecture

1. **Manifest discovery:** runtime scans only `infra_tools/` and the physically curated `registry/tools/`. Tool names, descriptions and schemas come from `tool.json`; no analytical implementation IDs are hard-coded in runtime code.
2. **Graph catalogue:** dataset IDs, descriptions, direction, weighting and available attributes are loaded dynamically so the planner never invents graph IDs or wastes calls listing known graphs.
3. **Semantic interpreter:** a dedicated structured Gemini call produces an immutable answer contract (type, cardinality, ordering, value meaning, required outputs and missing information) without keyword routing or tool planning.
4. **Planner:** Gemini receives the answer contract plus graph and manifest catalogues and submits an executable DAG of at most five graph-tool steps.
5. **Validation:** plans are rejected for unknown tools, cycles, bad dependencies, invalid bindings or manifest-incompatible arguments.
6. **Executor:** LangGraph resolves JSON Pointer bindings, runs ready steps sequentially against one `GraphStore`, validates outputs and preserves registered `result_graph_id` subgraphs.
7. **Evidence assessment:** a separate structured Gemini decision verifies that every answer-contract requirement is directly supported before finalization; insufficient evidence triggers bounded replanning.
8. **Finalization:** Gemini may answer only from successful evidence and must cite executed step IDs and output paths. Node, node-list and path values contain IDs rather than labels. Evidence remains in traces; answers retain the exact grader schema.
9. **Reliability:** transient Gemini failures use bounded exponential retry, successful responses are cached, answers are atomically persisted, and SQLite checkpoints support resume. In-memory subgraphs are reconstructed by replaying completed calls after restart.

## Traces

Each trace contains the question, every Gemini call or provider error, plan revisions, every tool call with implementation ID/arguments/output, replay events after resume, and the final structured answer. API credentials are never logged.

## Test

Build `registry/tools/` first (see "Curate Registry.zip"); most orchestration tests load the curated registry.

```bash
python -m pytest tests
python curate_registry.py verify
python smoke_test.py --no-llm
```

A live Gemini smoke run requires `LLM_API_KEY`.

## Project Layout

```text
curate_registry.py              offline candidate evaluation and materialization
run_queries.py                  required batch runner
orchestrator/registry.py        manifest-first strict discovery
orchestrator/graph_catalogue.py dynamic dataset metadata for planning
orchestrator/planner.py         Gemini DAG planning, evidence assessment and finalization
orchestrator/executor.py        bindings, tool calls and GraphStore replay
orchestrator/workflow.py        LangGraph state machine
orchestrator/provider.py        Gemini retries and cache
orchestrator/tracing.py         append-safe JSONL traces
registry/tools/                 curated one-implementation-per-tool registry (built from Registry.zip)
queries_public.json             official public question set
tests/project/                  orchestration and curation tests
```
