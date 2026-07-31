#!/usr/bin/env bash
# Re-run every measurement in TECHNICAL_NOTE.md from scratch.
#
#   ./scripts/run_experiments.sh
#
# Three runs, each against a freshly restarted app so conversation memory starts empty
# (level-2 history is process-local, and a stale chain would silently change the result):
#
#   1. shipped config      -> submission/                       the graded answers
#   2. REWRITE_ENABLED=0   -> docs/ablations/dropout-no-rewrite/  Level-2 ablation
#   3. AGENT_ENABLED=0     -> docs/ablations/dropout-no-agent/    Level-3 ablation
#
# Ablations write outside submission/ on purpose: they are meant to produce worse answers.
# .env is restored at the end whatever happens, so an interrupted run cannot leave the
# repository configured for an ablation.
set -uo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml"

restore() {
  sed -i '' 's/^REWRITE_ENABLED=.*/REWRITE_ENABLED=true/' .env
  sed -i '' 's/^AGENT_ENABLED=.*/AGENT_ENABLED=true/' .env
  echo "[restore] .env back to shipped defaults"
}
trap restore EXIT

restart() {
  $COMPOSE up -d --force-recreate app >/dev/null 2>&1
  for _ in $(seq 1 30); do
    curl -s -m 2 localhost:8791/health | grep -q ok && return 0
    sleep 2
  done
  echo "[error] app did not become healthy"; return 1
}

echo "=============== 1/3  shipped config -> submission/ ==============="
restore; restart || exit 1
python3 scripts/run_questions.py

echo
echo "=============== 2/3  REWRITE_ENABLED=false -> level 2 ==============="
sed -i '' 's/^REWRITE_ENABLED=.*/REWRITE_ENABLED=false/' .env
restart || exit 1
python3 scripts/run_questions.py 2 --out docs/ablations/dropout-no-rewrite

echo
echo "=============== 3/3  AGENT_ENABLED=false -> level 3 ==============="
sed -i '' 's/^REWRITE_ENABLED=.*/REWRITE_ENABLED=true/' .env
sed -i '' 's/^AGENT_ENABLED=.*/AGENT_ENABLED=false/' .env
restart || exit 1
python3 scripts/run_questions.py 3 --out docs/ablations/dropout-no-agent

echo
echo "=============== done — auditing the graded answers ==============="
restore; restart || exit 1
uv run python scripts/audit_quotes.py
