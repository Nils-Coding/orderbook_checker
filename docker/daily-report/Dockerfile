# Daily Report Generator for Cloud Run Jobs
# Runs once daily, reads from GCS, generates report, writes back to GCS

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY docker/daily-report/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tools/daily_report_light.py ./tools/
COPY docker/daily-report/run_report.sh .

# Make script executable
RUN chmod +x run_report.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the report script
ENTRYPOINT ["./run_report.sh"]
