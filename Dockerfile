FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Run Streamlit app (Render-compatible)
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
