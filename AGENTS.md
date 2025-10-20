# Repository Guidelines

## Project Structure & Module Organization
- `core/`: Orchestrates DSPy pipelines for assumption extraction, scoring, and exports.
- `contracts/`: Pydantic schemas; update here before changing downstream models.
- `plugins/` & `telemetry/`: LLM configuration, chart rendering, and instrumentation hooks.
- `service/` & `orchestration/`: FastAPI sidecar plus Prefect flow for hosted or batched runs; `run_direct.py` stays the CLI entry.
- `examples/`, `fixtures/`, `assets/`, `data/`: Sample payloads, reusable request bodies, chart outputs, and judge training data.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create and activate the local environment.
- `pip install -e .`: Install dependencies and the package in editable mode.
- `python run_direct.py examples/rehab_exercise_tracking_rich.json`: Execute a full JTBD analysis locally.
- `uvicorn service.dspy_sidecar:app --port 8088 --reload`: Serve REST endpoints for deconstruct, jobs, moat, and judge.

## Coding Style & Naming Conventions
- Target Python 3.10+, keep modules `snake_case`, and rely on type hints and Pydantic models for every boundary.
- Follow PEP 8 with 4-space indentation; favor small pure functions under `core/` and keep prompt text inside `dspy.Signature` docstrings.
- Store reusable integrations in `plugins/`, and prefer clear verb-noun function names such as `generate_jobs` or `merge_scores`.

## Testing Guidelines
- Adopt `pytest` with files under `tests/` mirroring `core/` structure (e.g., `tests/test_pipeline.py`).
- Seed tests with `fixtures/idea.json` or `examples/` payloads; mock network calls and set `JTBD_DSPY_MODEL` in the test environment.
- Run `pytest` before submitting and document coverage expectations for new flows or exporters.

## Commit & Pull Request Guidelines
- Write imperative, scope-focused commits similar to “Fix JudgeScoreSig to generate actual scores”.
- PRs should outline intent, touched directories, and validation performed (CLI run, API smoke test); attach screenshots only for chart changes.
- Link issues, note required secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), and request maintainer review when touching `contracts/` or orchestration.

## Security & Configuration Tips
- Manage API keys through environment variables listed in `README.md`; never commit `.env` files or generated artifacts under `assets/`.
- Keep new integrations inside `plugins/` to centralize rate limiting and credential handling.
