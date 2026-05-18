"""
main.py
FastAPI server for ProtAgent Phase 2.
"""

import asyncio
import threading
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import run as run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnotateRequest(BaseModel):
    fasta: str


class AnnotateResponse(BaseModel):
    job_id: str
    status: str


_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set_job(job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "error": error,
        }


def _get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def _run_pipeline(job_id: str, fasta: str) -> None:
    _set_job(job_id, "running")
    try:
        result = run_pipeline(fasta)
        _set_job(job_id, "complete", result=result)
    except Exception as exc:
        _set_job(job_id, "failed", error=str(exc))


@app.post("/annotate", response_model=AnnotateResponse, status_code=202)
async def annotate(request: AnnotateRequest) -> AnnotateResponse:
    job_id = str(uuid4())
    _set_job(job_id, "pending")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline, job_id, request.fasta)

    return AnnotateResponse(job_id=job_id, status="pending")


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, "status": job["status"]}


@app.get("/result/{job_id}")
async def result(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        return {"error": "not found"}
    if job["status"] != "complete":
        return {"error": "not ready"}
    return {"job_id": job_id, "status": job["status"], "result": job["result"]}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "phase2"}
