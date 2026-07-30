# Docling parser

Docling is the default parser. It runs as a local Docker service and is called by `/ingest`.

Pipeline:

```text
data/in/document.pdf -> POST /ingest -> Docling Serve -> chunks -> embeddings -> Qdrant -> POST /query -> data/out
```

Start the stack:

```powershell
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml up -d
```

Pull the Ollama chat model once:

```powershell
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec ollama ollama pull qwen3.6
docker compose -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.docling.yml -f docker-compose.reranker.yml exec ollama ollama pull bge-m3
```

Then import the Postman collection and run:

```text
Health -> Ready -> Ingest the PDF -> Query
```

Relevant settings:

```env
PDF_PARSER=docling
DOCLING_BASE_URL=http://localhost:5001
DOCLING_TABLE_MODE=accurate
DOCLING_IMAGE_EXPORT_MODE=referenced
DOCLING_DO_OCR=false
DOCLING_USE_CACHE=true
```

`/ingest` caches Docling JSON/Markdown and referenced image files under
`data/extracted/docling/<document>/`. Raw image bytes are saved separately rather than embedded into
retrieval chunks. These files are ignored by git because they can be regenerated from the committed
PDF. If you want to force a fresh parse, delete that folder before calling `/ingest`.

Docling is the only active parser in this project.
