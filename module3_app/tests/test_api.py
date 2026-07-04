"""
API tests for the Writing Studio — run entirely in-process via FastAPI's TestClient.
No network, no model download: they exercise the Stub backend (the default on a fresh clone)
and assert the contract + the privacy-relevant properties.

Run:  ../.venv/bin/python -m pytest module3_app/tests -q     # from repo root
"""

import os
import re

from fastapi.testclient import TestClient

from module3_app.backend.app import app

client = TestClient(app)
FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


def test_health_reports_local_backend():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "backend" in body and "is_real_model" in body


def test_rewrite_transforms_to_elizabethan():
    r = client.post("/rewrite", json={"text": "you are very tired", "intensity": 0.8})
    assert r.status_code == 200
    out = r.json()["output"].lower()
    # The stub deterministically swaps 2nd-person modern -> archaic forms.
    assert "thou art" in out and "most" in out


def test_rewrite_validates_input():
    assert client.post("/rewrite", json={"text": "", "intensity": 0.5}).status_code == 422
    assert client.post("/rewrite", json={"text": "hi", "intensity": 5}).status_code == 422


def test_feedback_is_in_character():
    r = client.post("/feedback", json={"text": "The weather is nice."})
    assert r.status_code == 200
    assert len(r.json()["output"]) > 20


def test_chat_responds():
    r = client.post("/chat", json={"message": "Well met, Bard."})
    assert r.status_code == 200
    assert r.json()["output"]


def test_frontend_has_no_external_asset_urls():
    """The no-egress guarantee starts here: the shipped frontend must reference no remote host."""
    offenders = []
    for root, _dirs, files in os.walk(FRONTEND):
        for fn in files:
            if not fn.endswith((".html", ".js", ".css")):
                continue
            text = open(os.path.join(root, fn), encoding="utf-8").read()
            # Any absolute URL to a non-loopback host is forbidden.
            for m in re.findall(r"https?://[^\s\"')]+", text):
                host = re.sub(r"https?://", "", m).split("/")[0].split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    offenders.append(f"{fn}: {m}")
    assert not offenders, f"external asset URLs found: {offenders}"
