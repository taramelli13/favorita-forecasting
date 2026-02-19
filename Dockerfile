FROM python:3.13-slim

RUN pip install --no-cache-dir poetry==2.3.2

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

# Copy source code
COPY src/ src/
COPY params.yaml configs/ ./
RUN poetry install --only-root --no-interaction

ENTRYPOINT ["poetry", "run", "favorita"]
CMD ["--help"]
