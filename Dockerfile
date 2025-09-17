FROM python:3.8-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src
COPY templates ./templates
COPY app.py main.py ./
COPY config ./config
COPY params.yaml dvc.yaml ./

#Copy model
COPY model ./model

EXPOSE 8080

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Start app with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "180"]

