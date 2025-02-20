# Use the official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port (8000) and frontend port (8501 for Streamlit)
EXPOSE 8000 8501

# Default command to run the FastAPI API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
