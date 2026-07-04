#!/usr/bin/env bash
# no_egress_test.sh — prove the Writing Studio makes ZERO external network connections.
#
# It starts the backend, hammers every endpoint (rewrite/feedback/chat), and inspects the
# server process's live sockets with lsof. Any connection whose FOREIGN address is not
# loopback (127.0.0.1 / ::1 / localhost) fails the test. A LISTEN socket on 127.0.0.1 and
# loopback->loopback connections from the browser/curl are expected and allowed.
#
# Usage:  ./no_egress_test.sh            # starts its own server on 127.0.0.1:8899
# Exit:   0 = no egress (pass), 1 = external connection detected (fail)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
PORT="${BARDBOX_PORT:-8899}"
export BARDBOX_PORT="$PORT"

echo "→ starting backend on 127.0.0.1:$PORT"
"$PY" -m module3_app.backend.app >/tmp/bardbox_egress_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# wait for /health
for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.25
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "✗ server did not come up — see /tmp/bardbox_egress_server.log"; exit 1
fi

echo "→ exercising endpoints (rewrite, feedback, chat)"
curl -sf -X POST "http://127.0.0.1:$PORT/rewrite"  -H 'Content-Type: application/json' -d '{"text":"I am very tired.","intensity":0.7}' >/dev/null
curl -sf -X POST "http://127.0.0.1:$PORT/feedback" -H 'Content-Type: application/json' -d '{"text":"Please stop lying to me."}' >/dev/null
curl -sf -X POST "http://127.0.0.1:$PORT/chat"     -H 'Content-Type: application/json' -d '{"message":"Well met, Bard."}' >/dev/null

echo "→ inspecting the server process sockets (pid $SRV)"
# List this process's network files; keep only real connections (has a '->' foreign addr).
# NOTE: -a ANDs the -p and -i selectors. Without it lsof ORs them and dumps EVERY
# internet connection on the machine (a classic lsof footgun → false positives).
CONNS="$(lsof -nP -a -p "$SRV" -i 2>/dev/null | awk '/->/{print $9}')"

EXTERNAL=""
while IFS= read -r c; do
  [ -z "$c" ] && continue
  foreign="${c##*->}"          # strip local side, keep foreign endpoint host:port
  host="${foreign%:*}"          # drop :port
  case "$host" in
    127.0.0.1|::1|localhost|\[::1\]) : ;;   # loopback = allowed
    *) EXTERNAL="$EXTERNAL$host\n" ;;
  esac
done <<< "$CONNS"

echo "-----------------------------------------------------------"
if [ -z "$EXTERNAL" ]; then
  echo "✓ PASS — no egress. Every connection stayed on loopback."
  echo "  The Writing Studio talked to nothing but your own machine."
  exit 0
else
  echo "✗ FAIL — external connection(s) detected:"
  printf "  %b" "$EXTERNAL"
  exit 1
fi
