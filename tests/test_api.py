import asyncio

import pytest
from httpx import AsyncClient, ASGITransport

import main


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "phase2"}


@pytest.mark.asyncio
async def test_annotate_and_status_and_result(monkeypatch):
    with main._lock:
        main._jobs.clear()

    def fake_run_pipeline(job_id: str, fasta: str) -> None:
        main._set_job(job_id, "complete", result={"ok": True})

    monkeypatch.setattr(main, "_run_pipeline", fake_run_pipeline)

    async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as client:
        resp = await client.post("/annotate", json={"fasta": ">seq\nAAAA"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(10):
            status_resp = await client.get(f"/status/{job_id}")
            status = status_resp.json()["status"]
            if status == "complete":
                break
            await asyncio.sleep(0.01)

        result_resp = await client.get(f"/result/{job_id}")
        assert result_resp.status_code == 200
        assert result_resp.json()["status"] in ("complete",)
