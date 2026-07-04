# Act III — The Private Shakespearean Writing Studio

A fully-offline desktop-style web app that runs the fine-tuned model **on your own machine**
and rewrites your drafts in Shakespeare's voice — with a **provable no-egress guarantee**. This
is the "I can ship a model that protects user data" act.

> **Why the privacy angle is real:** your unpublished creative writing is sensitive intellectual
> property. A cloud writing assistant may log, retain, or train on what you send it. Bardbox never
> makes that trade — the model is a file on your disk, and the app has no network code path to
> anywhere but your own machine.

## Run it

```bash
# from the repo root, in the project venv
.venv/bin/python -m module3_app.backend.app
# open http://127.0.0.1:8000
```

That's it. On a fresh clone with no model, it runs in **demo mode** (a built-in rule-based
transformer) so the whole app and the privacy test work immediately. To use the **real** model,
drop a GGUF into `module3_app/models/` (see `../module2_finetune/merge_and_quantize.md`) and
restart — it's picked up automatically.

## The three views (see `../VIEWS.md`)

- **Studio** — two panes: your modern draft (left) → the Bard's rewrite (right, EB Garamond),
  with a Subtle↔Full-Elizabethan intensity slider and a "Ground in the works" citation toggle.
- **Feedback** — the right pane becomes in-character critique of your draft.
- **Privacy** — a live "0 outbound connections" monitor, a self-test button, and the
  airplane-mode challenge. Plus **Chat** and **Settings**.

## How the privacy guarantee is built (not just asserted)

1. **Localhost-only.** The server binds to `127.0.0.1` (never `0.0.0.0`) — unreachable off-machine.
2. **No remote backend exists.** `backend/inference.py` has exactly three backends — an in-process
   GGUF, a `127.0.0.1` llama.cpp/Ollama server, and a pure-Python stub. None can reach a remote host.
3. **No external front-end assets.** All CSS/JS and the OFL EB Garamond font are served from this
   same localhost origin. A test greps the shipped frontend for any external URL and fails if found.
4. **A live no-egress test.** `privacy/no_egress_test.sh` starts the server, drives every endpoint,
   and inspects the process's sockets with `lsof` — failing if any connection leaves loopback.

```bash
./module3_app/privacy/no_egress_test.sh     # → "✓ PASS — no egress"
.venv/bin/python -m pytest module3_app/tests -q
```

See `privacy/airplane_mode_checklist.md` for the 60-second manual proof.

## Layout

```
module3_app/
├── backend/
│   ├── app.py         # FastAPI (127.0.0.1), endpoints: /health /rewrite /feedback /chat
│   ├── inference.py   # pluggable backends: in-process GGUF → local server → stub
│   └── prompts.py     # task prompts + intensity→temperature mapping
├── frontend/
│   ├── index.html     # single bundled page (Studio/Feedback/Chat/Privacy/Settings)
│   └── assets/        # app.js + the OFL EB Garamond font — all local, no CDN
├── privacy/           # no_egress_test.sh + airplane_mode_checklist.md
├── models/            # drop a .gguf here to use the real model (git-ignored)
└── tests/             # pytest: endpoint contract + no-external-asset-URLs check
```
