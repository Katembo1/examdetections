# syntax=docker/dockerfile:1
FROM python:3.12-slim

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1000 --retries 10 -r /app/requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "run.py"]
