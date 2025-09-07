from prefect import flow, task
from contracts.idea_v1 import IdeaV1
from core.pipeline import run_pipeline
from pathlib import Path

@task(retries=2, retry_delay_seconds=2)
def ensure_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

@task(retries=1)
def run_stage(idea: IdeaV1, assets_dir: str):
    return run_pipeline(idea, assets_dir)

@flow(name="jtbd-report-dspy")
def jtbd_report(idea: IdeaV1, out_md: str, assets_dir: str):
    ensure_dirs.submit(assets_dir)
    result = run_stage(idea, assets_dir)
    Path(out_md).write_text(result["gamma_md"], encoding="utf-8")
    return result
