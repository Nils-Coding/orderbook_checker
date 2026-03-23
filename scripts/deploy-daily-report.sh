#!/bin/bash
# Deploy Daily Report as Cloud Run Job
# Runs once daily via Cloud Scheduler

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-orderflow-recorder}"
REGION="${REGION:-europe-west1}"
JOB_NAME="daily-report"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}"
SCHEDULER_NAME="trigger-daily-report"
GCS_BUCKET="gs://orderflow-data-lake/orderbook-checker"
SERVICE_ACCOUNT="orderbook-recorder@${PROJECT_ID}.iam.gserviceaccount.com"

echo "========================================"
echo "Deploying Daily Report Cloud Run Job"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo "========================================"

# Change to project root
cd "$(dirname "$0")/.."

# Enable required APIs
echo ""
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com --project="${PROJECT_ID}"
gcloud services enable cloudscheduler.googleapis.com --project="${PROJECT_ID}"
gcloud services enable cloudbuild.googleapis.com --project="${PROJECT_ID}"

# Build and push Docker image
echo ""
echo "Building Docker image..."
gcloud builds submit . \
    --project="${PROJECT_ID}" \
    --config=docker/daily-report/cloudbuild.yaml

# Create/Update Cloud Run Job
echo ""
echo "Creating Cloud Run Job..."

# Check if job exists
if gcloud run jobs describe "${JOB_NAME}" --project="${PROJECT_ID}" --region="${REGION}" &>/dev/null; then
    echo "Job exists, updating..."
    gcloud run jobs update "${JOB_NAME}" \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --image="${IMAGE_NAME}" \
        --memory="2Gi" \
        --cpu="1" \
        --max-retries=1 \
        --task-timeout="30m" \
        --set-env-vars="GCS_BUCKET=${GCS_BUCKET}" \
        --service-account="${SERVICE_ACCOUNT}"
else
    echo "Creating new job..."
    gcloud run jobs create "${JOB_NAME}" \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --image="${IMAGE_NAME}" \
        --memory="2Gi" \
        --cpu="1" \
        --max-retries=1 \
        --task-timeout="30m" \
        --set-env-vars="GCS_BUCKET=${GCS_BUCKET}" \
        --service-account="${SERVICE_ACCOUNT}"
fi

# Create Cloud Scheduler job to trigger daily at 04:00 UTC
# (Later than before to allow full day of data, away from peak hours)
echo ""
echo "Creating Cloud Scheduler trigger..."

# Check if scheduler job exists
if gcloud scheduler jobs describe "${SCHEDULER_NAME}" --project="${PROJECT_ID}" --location="${REGION}" &>/dev/null; then
    echo "Scheduler exists, updating..."
    gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --schedule="0 4 * * *" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email="${SERVICE_ACCOUNT}" \
        --description="Trigger daily report generation at 04:00 UTC"
else
    echo "Creating new scheduler..."
    gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --schedule="0 4 * * *" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email="${SERVICE_ACCOUNT}" \
        --description="Trigger daily report generation at 04:00 UTC"
fi

echo ""
echo "========================================"
echo "Deployment complete!"
echo ""
echo "Cloud Run Job: ${JOB_NAME}"
echo "Scheduler: ${SCHEDULER_NAME} (04:00 UTC daily)"
echo ""
echo "To run manually:"
echo "  gcloud run jobs execute ${JOB_NAME} --project=${PROJECT_ID} --region=${REGION}"
echo ""
echo "To check logs:"
echo "  gcloud run jobs logs ${JOB_NAME} --project=${PROJECT_ID} --region=${REGION}"
echo "========================================"
