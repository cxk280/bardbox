# Airplane-mode checklist — prove it to yourself in 60 seconds

The strongest privacy demo needs no tools: **disconnect, and watch nothing break.**

1. **Start the app while online** (so any first-run downloads, if you added a real model, are done):
   `../.venv/bin/python -m module3_app.backend.app` and open http://127.0.0.1:8000.
2. **Turn off Wi-Fi / pull the ethernet cable / enable Airplane Mode.**
3. **Keep working.** Rewrite a draft, ask for notes, chat with the Bard. Everything responds
   exactly as before — because every byte is computed on your machine.
4. **Open the Privacy view** and click *Run privacy self-test*: it confirms the backend is on-device.
5. For a rigorous, automated proof, run `./no_egress_test.sh` — it drives every endpoint and
   inspects the server's live sockets, failing if a single connection leaves loopback.

## What "no egress" rests on (the guarantees behind the demo)

- The server binds to **127.0.0.1 only** — unreachable from any other machine.
- The frontend loads **no external assets** — the bundled OFL font and all CSS/JS are served
  from this same localhost origin (verified by a grep in the test suite).
- The only inference backends that exist are **in-process** (a local `.gguf`) or a **127.0.0.1**
  server. There is no code path to a remote host.
- **No telemetry, no analytics, no accounts.** Nothing to phone home with.
