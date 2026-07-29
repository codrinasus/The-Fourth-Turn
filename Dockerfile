# The Fourth Turn — app image.
# Build context is the repo root; docker-compose.yml builds this for you.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# uv is the package manager (https://docs.astral.sh/uv). Never use pip here.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first, in their own layer, so editing app code does not
# re-resolve the whole dependency tree on every rebuild.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --no-cache

# Now the application code.
COPY app ./app
RUN uv sync --frozen --no-dev --no-cache

EXPOSE 8000

# uvicorn serves the FastAPI app. --reload is deliberately NOT used in the image;
# add it in docker-compose if you want hot reload while developing.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
