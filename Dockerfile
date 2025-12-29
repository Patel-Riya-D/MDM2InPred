FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
