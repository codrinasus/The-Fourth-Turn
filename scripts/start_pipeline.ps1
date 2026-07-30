param(
    [string]$Model = "qwen3.6",
    [string]$EmbeddingModel = "bge-m3",
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location $Root

$compose = @(
    "compose",
    "-f", "docker-compose.yml",
    "-f", "docker-compose.ollama.yml",
    "-f", "docker-compose.docling.yml",
    "-f", "docker-compose.reranker.yml"
)

docker @compose up -d ollama qdrant docling reranker app

if (-not $SkipModelPull) {
    docker @compose exec -T ollama ollama pull $Model
    docker @compose exec -T ollama ollama pull $EmbeddingModel
}

Write-Host "Pipeline services started."
Write-Host "Postman order: Health -> Ready -> Ingest the PDF -> Query."
Write-Host "Swagger: http://localhost:8791/docs"
Write-Host "Docling: http://localhost:5001/docs"
Write-Host "Rerank:  http://localhost:8792/health   (first start downloads ~600 MB of GGUF)"
Write-Host "Qdrant:  http://localhost:6391/dashboard"
