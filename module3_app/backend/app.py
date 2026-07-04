"""
The Writing Studio backend — FastAPI, bound to localhost, serving a bundled frontend.

Security posture: this process only ever talks to (a) an in-process model or (b) a
127.0.0.1 inference server. It binds to 127.0.0.1 (never 0.0.0.0) so it isn't reachable
off-machine, and the frontend it serves loads zero external assets. The `no_egress_test`
in ../privacy/ verifies the "no outbound connections" claim against the running process.

Run:
    python -m module3_app.backend.app       # from repo root
    # or: cd module3_app && ../.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference import select_backend

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(os.path.dirname(HERE), "frontend")

app = FastAPI(title="Bardbox — Writing Studio", docs_url=None, redoc_url=None)

# Pick the backend once at import/startup (the composition root lives in inference.py).
BACKEND = select_backend()


class RewriteReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    intensity: float = Field(0.5, ge=0.0, le=1.0)


class TextReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class ChatReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


def _meta() -> dict:
    return {"backend": BACKEND.name, "is_real_model": BACKEND.is_real_model}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **_meta()}


@app.post("/rewrite")
def rewrite(req: RewriteReq) -> dict:
    from .prompts import SYSTEM_REWRITE, intensity_to_temperature, rewrite_instruction

    system = SYSTEM_REWRITE + rewrite_instruction(req.intensity)
    out = BACKEND.generate(system, req.text, max_tokens=220, temperature=intensity_to_temperature(req.intensity))
    return {"output": out, **_meta()}


@app.post("/feedback")
def feedback(req: TextReq) -> dict:
    from .prompts import SYSTEM_FEEDBACK

    out = BACKEND.generate(SYSTEM_FEEDBACK, req.text, max_tokens=220, temperature=0.7)
    return {"output": out, **_meta()}


@app.post("/chat")
def chat(req: ChatReq) -> dict:
    from .prompts import SYSTEM_CHAT

    out = BACKEND.generate(SYSTEM_CHAT, req.message, max_tokens=220, temperature=0.8)
    return {"output": out, **_meta()}


# Serve the bundled single-page frontend. All assets are local (see frontend/).
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND, "assets")), name="assets")


def main() -> None:
    import uvicorn

    # 127.0.0.1 ONLY — not reachable from other machines, part of the privacy posture.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BARDBOX_PORT", "8000")))


if __name__ == "__main__":
    main()
