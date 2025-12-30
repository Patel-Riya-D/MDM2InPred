FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Java for padelpy)
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Render provides PORT env variable
ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
