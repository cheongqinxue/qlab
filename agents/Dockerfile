# Shared agents runtime — deps + agents/ (build once, reuse per experiment).
# Build: agents/build-image.sh

FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev

COPY agents /app/agents

ENV PYTHONPATH=/app/agents
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app/agents

# Experiments override command + working_dir in compose.
CMD ["python", "-c", "print('agents-runtime image ready; set experiment command in compose')"]
