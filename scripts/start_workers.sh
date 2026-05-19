#!/usr/bin/env bash
# Start the 10-worker TRIBE stack:
#   external ngrok daemon  ->  caddy :7860  ->  10 uvicorn workers :7861-7870
# 5 workers pinned to GPU 0, 5 to GPU 1. Each worker has its own MODEL + lock.
#
# Usage:
#   ./scripts/start_workers.sh          # start everything
#   ./scripts/start_workers.sh stop     # stop everything

set -uo pipefail

APP_DIR="/home/ripper/ACTi_cognition"
BASE_PORT=7861
NUM_WORKERS=10
WORKERS_PER_GPU=5
RUN_DIR="/tmp/tribe-caddy"
LOG_DIR="/tmp/tribe-workers"
CADDY="/home/ripper/bin/caddy"
NGROK_BIN="$(command -v ngrok || echo /snap/bin/ngrok)"

mkdir -p "$RUN_DIR" "$LOG_DIR"

stop_all() {
	echo "=== stopping TRIBE stack ==="
	# Stop external ngrok tunnel for acti.cognition
	pkill -f "ngrok.*acti.cognition.ngrok.pro" 2>/dev/null && echo "  ngrok stopped"
	# Stop caddy (match the actual `caddy run --config .../scripts/Caddyfile` cmdline)
	pkill -f "caddy run.*scripts/Caddyfile" 2>/dev/null && echo "  caddy stopped"
	# Stop workers (every python3 app.py on this box)
	pkill -f "python3 app.py" 2>/dev/null && echo "  workers stopped"
	sleep 2
	echo "=== done ==="
}

if [[ "${1:-}" == "stop" ]]; then
	stop_all
	exit 0
fi

cd "$APP_DIR"

echo "=== stopping any existing TRIBE / in-process tunnel ==="
pkill -f "python3 app.py" 2>/dev/null && echo "  killed old workers"
# The old in-process pyngrok will die with its parent process.
sleep 2

# Free the public ngrok domain if a stale daemon is holding it.
pkill -f "ngrok.*acti.cognition.ngrok.pro" 2>/dev/null && echo "  killed stale ngrok daemon"
sleep 1

source venv/bin/activate

echo "=== starting external ngrok daemon (acti.cognition.ngrok.pro -> :7860) ==="
NGROK_AUTHTOKEN="$(grep -E '^NGROK_AUTHTOKEN=' .env | cut -d= -f2-)"
NGROK_DOMAIN="$(grep -E '^NGROK_DOMAIN=' .env | cut -d= -f2-)"
NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" nohup "$NGROK_BIN" http 7860 \
	--domain="$NGROK_DOMAIN" \
	--log=stdout --log-format=logfmt \
	> "$LOG_DIR/ngrok.log" 2>&1 &
NGROK_PID=$!
echo "  ngrok pid $NGROK_PID -> https://$NGROK_DOMAIN"

echo "=== starting caddy reverse proxy on :7860 ==="
nohup "$CADDY" run \
	--config "$APP_DIR/scripts/Caddyfile" \
	--adapter caddyfile \
	> "$LOG_DIR/caddy.log" 2>&1 &
CADDY_PID=$!
echo "  caddy pid $CADDY_PID"
sleep 1

echo "=== starting $NUM_WORKERS TRIBE workers ==="
for i in $(seq 0 $((NUM_WORKERS - 1))); do
	port=$((BASE_PORT + i))
	gpu=$((i / WORKERS_PER_GPU))  # 0..4 -> GPU 0,  5..9 -> GPU 1
	CUDA_VISIBLE_DEVICES="$gpu" \
	PORT="$port" \
	NGROK_DISABLE=1 \
		nohup python3 app.py > "$LOG_DIR/worker-$port.log" 2>&1 &
	echo "  worker pid $! port $port gpu $gpu"
done

echo
echo "=== waiting for all workers to become healthy (model load) ==="
deadline=$(( $(date +%s) + 180 ))
while [[ $(date +%s) -lt $deadline ]]; do
	ready=0
	for i in $(seq 0 $((NUM_WORKERS - 1))); do
		port=$((BASE_PORT + i))
		if curl -s --max-time 2 "http://127.0.0.1:$port/healthz" 2>/dev/null \
			| grep -q '"model_loaded":true'; then
			ready=$((ready + 1))
		fi
	done
	echo "  $ready / $NUM_WORKERS workers healthy"
	if [[ $ready -eq $NUM_WORKERS ]]; then
		break
	fi
	sleep 5
done

echo
echo "=== final state ==="
echo "Public URL:   https://$NGROK_DOMAIN"
echo "Auth:         user / 67420"
echo "Worker logs:  $LOG_DIR/worker-*.log"
echo "Caddy log:    $LOG_DIR/caddy.log"
echo "ngrok log:    $LOG_DIR/ngrok.log"
