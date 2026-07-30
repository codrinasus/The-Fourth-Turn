# Start The Pipeline

The pipeline keeps the challenge input/output contract:

```text
data/in/document.pdf -> POST /ingest -> Docling extraction -> Qdrant -> POST /query -> data/out
```

Docling is integrated at `/ingest`: the app sends the PDF in `data/in/` to the local Docling
service, caches structured artifacts, indexes pages/chunks in Qdrant, and records
`parser=docling` in each payload.

## Windows

From the repository root:

```powershell
Copy-Item .env.example .env
.\scripts\start_pipeline.ps1
```

Useful options:

```powershell
.\scripts\start_pipeline.ps1 -SkipModelPull
```

## macOS

From the repository root:

```bash
cp .env.example .env
bash scripts/start_pipeline.sh
```

Useful options:

```bash
SKIP_MODEL_PULL=true bash scripts/start_pipeline.sh
```

## Manual Steps

Windows:

```powershell
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml up -d ollama qdrant docling reranker app
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec ollama ollama pull qwen3.6
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec ollama ollama pull bge-m3
curl.exe http://localhost:8791/ingest -H "content-type: application/json" -d "{}"
```

macOS:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml up -d ollama qdrant docling reranker app
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec -T ollama ollama pull qwen3.6
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec -T ollama ollama pull bge-m3
curl -fsS -X POST http://localhost:8791/ingest -H "content-type: application/json" -d "{}"
```

After startup:

- Swagger: `http://localhost:8791/docs`
- Docling API: `http://localhost:5001/docs`
- Reranker: `http://localhost:8792/health` (bge-reranker-v2-m3 via llama.cpp `/v1/rerank`)
- Qdrant dashboard: `http://localhost:6391/dashboard`
- Query endpoint: `POST http://localhost:8791/query`

Level-2 reminder: send q4, q5, q6 in order with `level: 2`; the system threads them
automatically. You do not pass a conversation id.
