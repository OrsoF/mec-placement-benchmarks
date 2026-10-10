FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
ARG MECBENCH_EXTRAS=benchmarks
RUN python -m pip install --no-cache-dir ".[${MECBENCH_EXTRAS}]"
COPY data/ ./data/

ENTRYPOINT ["mecbench"]
CMD ["--help"]
