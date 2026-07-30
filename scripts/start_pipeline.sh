#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-qwen3.6}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
SKIP_MODEL_PULL="${SKIP_MODEL_PULL:-false}"

compose=(
  docker compose
  -f docker-compose.yml
  -f docker-compose.ollama.yml
  -f docker-compose.docling.yml
  -f docker-compose.reranker.yml
)

"${compose[@]}" up -d ollama qdrant docling reranker app

if [[ "$SKIP_MODEL_PULL" != "true" ]]; then
  "${compose[@]}" exec -T ollama ollama pull "$MODEL"
  "${compose[@]}" exec -T ollama ollama pull "$EMBEDDING_MODEL"
fi

echo "Pipeline services started."
echo "Postman order: Health -> Ready -> Ingest the PDF -> Query."
echo "Swagger: http://localhost:8791/docs"
echo "Docling: http://localhost:5001/docs"
echo "Rerank:  http://localhost:8792/health   (first start downloads ~600 MB of GGUF)"
echo "Qdrant:  http://localhost:6391/dashboard"
