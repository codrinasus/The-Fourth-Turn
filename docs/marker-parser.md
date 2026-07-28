# Marker Parser

Marker is the parser for this document. We tested it against GROBID and kept Marker because it
preserved page separators, headings, and table blocks better for the committed PDF.

The challenge architecture stays intact:

```text
data/in/document.pdf -> Marker output -> POST /ingest -> Qdrant -> POST /query -> data/out
```

Run Marker before ingesting:

```powershell
.\scripts\run_marker.ps1
.\scripts\run_marker.ps1 -Format json -OutDir data/extracted/marker-json
```

Then start the backend and ingest normally:

```powershell
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
curl.exe http://localhost:8791/ingest -H "content-type: application/json" -d "{}"
```

`/ingest` reads `data/extracted/marker/<document>/<document>.md` first and falls back to
`data/extracted/marker-json/<document>/<document>.json`. If neither exists, it fails with a clear
message instead of silently falling back to lower-quality `pypdf` extraction.

Marker is run in CPU-friendly `fast --disable_ocr` mode. OCR is intentionally off in this image
because Marker 2's OCR path requires a `llama-server` binary. For this digital PDF, test the text
layer first.
