param(
    [string]$Pdf = "data/in/document.pdf",
    [string]$OutDir = "data/extracted/marker",
    [ValidateSet("markdown", "json", "html", "chunks")]
    [string]$Format = "markdown"
)

$ErrorActionPreference = "Stop"

$argsForMarker = @(
    $Pdf,
    "--mode", "fast",
    "--output_dir", $OutDir,
    "--output_format", $Format
)

if ($Format -eq "markdown") {
    $argsForMarker += "--MarkdownRenderer_paginate_output"
}

$argsForMarker += "--disable_ocr"

docker compose -f docker-compose.marker.yml build marker
docker compose -f docker-compose.marker.yml run --rm marker @argsForMarker
