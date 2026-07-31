#!/usr/bin/env bash
# Set a boolean setting in .env and prove the running container picked it up.
#
#   ./scripts/set_flag.sh REWRITE_ENABLED false
#
# Editing .env and recreating is not enough on its own. Docker Desktop syncs the bind
# mount through a VM, so `docker compose up --force-recreate` immediately after a write
# can recreate the container from the *previous* contents — we lost an ablation to exactly
# that, and the run silently changed two variables instead of one. An ablation you cannot
# prove ran is worse than no ablation, so this verifies and retries.
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="$1"; VALUE="$2"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml"

sed -i '' "s/^${NAME}=.*/${NAME}=${VALUE}/" .env
grep -q "^${NAME}=${VALUE}$" .env || { echo "[set_flag] .env edit failed for ${NAME}"; exit 1; }

for attempt in 1 2 3; do
  sleep 2   # let the file reach the VM before compose reads it
  $COMPOSE up -d --force-recreate app >/dev/null 2>&1
  until curl -s -m 2 localhost:8791/health | grep -q ok; do sleep 2; done
  actual=$($COMPOSE exec -T app sh -c "echo \$${NAME}" | tr -d '\r\n')
  if [ "$actual" = "$VALUE" ]; then
    echo "[set_flag] ${NAME}=${actual} (attempt ${attempt})"
    exit 0
  fi
  echo "[set_flag] attempt ${attempt}: container still has ${NAME}=${actual}, retrying"
done

echo "[set_flag] FAILED to apply ${NAME}=${VALUE}"; exit 1
