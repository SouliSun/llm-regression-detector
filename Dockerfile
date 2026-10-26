FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GROQ_API_KEY="" \
    OPENAI_API_KEY="" \
    SLACK_WEBHOOK_URL="" \
    MAX_REGRESSIONS=0 \
    MIN_DELTA=0.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY evals/ ./evals/
COPY data/ ./data/
COPY prompts/ ./prompts/
COPY results/ ./results/
COPY pytest.ini .

CMD ["python", "-m", "evals.run", "--prompt", "v1"]