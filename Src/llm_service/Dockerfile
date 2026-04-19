FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better Docker caching)
COPY requirements.txt .
RUN uv add --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Never run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]