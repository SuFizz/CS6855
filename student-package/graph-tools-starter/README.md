# Group 2 Graph Analytics Orchestrator

A Gemini-powered LangGraph orchestrator that plans, executes and chains manifest-discovered graph tools while grounding every answer in traced tool evidence.

## Requirements

- Python 3.10 or newer
- Google Gemini with OpenAI-compatible function calling
- `LLM_API_KEY`
- The published `Registry.zip`

The exact Gemini model remains configurable with `LLM_MODEL`. Never commit an API key.

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

## Curate Registry.zip

Extract `Registry.zip` in the top-level Assignment 3 directory so its implementations are under `tools/<tool_name>/<implementation_id>/`. The runtime does not choose between duplicate student implementations. Curation independently tests candidates and materializes exactly one implementation per official tool name.

From `graph-tools-starter`:

```bash
python curate_registry.py evaluate \
  --source ../../tools \
  --report .orchestrator/curation_report.json \
  --selection .orchestrator/registry_selection.json
```

Review or override `.orchestrator/registry_selection.json`, then create the final registry:

```bash
python curate_registry.py materialize \
  --source ../../tools \
  --selection .orchestrator/registry_selection.json \
  --dest registry/tools \
  --replace

python curate_registry.py verify
```

Registry source files are copied unchanged. Strict verification requires one implementation for all 33 analytical tools plus the four instructor infrastructure tools.

## Run

The required command remains:

```bash
python run_queries.py --queries queries_public.json --out answers_public.json
```

Traces are written to `traces/<query_id>.jsonl`. Checkpoints and Gemini responses are stored in the ignored `.orchestrator/` directory.

Useful development options:

```bash
python run_queries.py \
  --queries queries_registry_all.json \
  --out answers_registry_all.json \
  --resume
```

- `--resume`: skip terminal answers and resume interrupted workflows
- `--retry-errors`: rerun prior error results when resuming
- `--no-cache`: disable Gemini response caching
- `--allow-incomplete-registry`: warn and continue instead of failing on missing/duplicate tools
- `--state-dir`: choose a different checkpoint/cache directory

Strict registry startup is the default. Warn mode is intended only for development.

## Query formats

The runner accepts plain text, JSON arrays, JSON objects containing a `queries` list, and JSONL. Objects may use `query_id`/`id` and `question`/`query`/`text`.

The included `queries_registry_all.json` is a 41-question development set covering all analytical and infrastructure tools, two chaining scenarios, ambiguity and an unsupported request. It is not the official public set.

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

```bash
python -m pytest tests
python curate_registry.py verify
python smoke_test.py --no-llm
```

A live Gemini smoke run requires `LLM_API_KEY`.

Score a public run offline without exposing expected answers to the orchestrator:

```bash
python evaluation/score_public.py answers_public.json
# Or score a trace archive:
python evaluation/score_public.py ../../public_queries.zip
```

The evaluator checks status, answer type, numeric tolerance, node IDs, ranked order, paths and topological-order validity. Nothing under `evaluation/` is imported by runtime orchestration code.

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
registry/tools/                 generated one-implementation-per-tool registry
queries_registry_all.json       non-public development questions
evaluation/                     offline public expected answers and scorer
tests/project/                  orchestration and curation tests
```
