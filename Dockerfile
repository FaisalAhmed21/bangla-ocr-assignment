FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install packages first so Docker can cache this slow layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app

COPY --chown=app:app app.py common.py segmentation.py prediction.py config.py labels.json ./
COPY --chown=app:app models ./models

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
