FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp/jiuwen-cache \
    HF_HOME=/tmp/jiuwen-cache/huggingface \
    MPLCONFIGDIR=/tmp/jiuwen-cache/matplotlib \
    TIKTOKEN_CACHE_DIR=/tmp/jiuwen-cache/tiktoken \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

COPY run_agent.py .
COPY harness ./harness

RUN python -m compileall -q /app /usr/local/lib/python3.11/site-packages \
    && python -m harness.smoke
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
