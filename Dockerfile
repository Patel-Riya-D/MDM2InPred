FROM python:3.10-slim

# Install system dependencies (Java 11 + build tools)
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Streamlit port
EXPOSE 7860

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
