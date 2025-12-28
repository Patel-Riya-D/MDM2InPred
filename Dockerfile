FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openjdk-17-jre \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# RDKit installation
RUN pip install --no-cache-dir rdkit-pypi

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 7860

# Streamlit command
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
