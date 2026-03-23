#!/bin/bash
# Daily Report Runner for Cloud Run Jobs
# Downloads data from GCS, generates report, uploads back to GCS

set -e

# Configuration (can be overridden via environment variables)
GCS_BUCKET="${GCS_BUCKET:-gs://orderflow-data-lake/orderbook-checker}"
SYMBOL="${SYMBOL:-BTCUSDT}"
REPORT_DATE="${REPORT_DATE:-$(date -u -d 'yesterday' +%Y-%m-%d)}"

echo "========================================"
echo "Daily Report Generator"
echo "Date: ${REPORT_DATE}"
echo "Symbol: ${SYMBOL}"
echo "Bucket: ${GCS_BUCKET}"
echo "========================================"

# Create working directories
WORK_DIR="/tmp/report_work"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/reports/${REPORT_DATE}"
mkdir -p "${DATA_DIR}/snapshots/${SYMBOL}/${REPORT_DATE}"
mkdir -p "${DATA_DIR}/trades/${SYMBOL}/${REPORT_DATE}"
mkdir -p "${OUTPUT_DIR}"

# Download data from GCS
echo ""
echo "Downloading snapshot data..."
gsutil -m cp -r "${GCS_BUCKET}/data/snapshots/${SYMBOL}/${REPORT_DATE}/" "${DATA_DIR}/snapshots/${SYMBOL}/${REPORT_DATE}/" || {
    echo "Warning: No snapshot data found for ${REPORT_DATE}"
}

echo ""
echo "Downloading trade data..."
gsutil -m cp -r "${GCS_BUCKET}/data/trades/${SYMBOL}/${REPORT_DATE}/" "${DATA_DIR}/trades/${SYMBOL}/${REPORT_DATE}/" || {
    echo "Warning: No trade data found for ${REPORT_DATE}"
}

# Count downloaded files
SNAP_COUNT=$(find "${DATA_DIR}/snapshots" -name "*.parquet" 2>/dev/null | wc -l)
TRADE_COUNT=$(find "${DATA_DIR}/trades" -name "*.parquet" 2>/dev/null | wc -l)
echo ""
echo "Downloaded: ${SNAP_COUNT} snapshot files, ${TRADE_COUNT} trade files"

if [ "$SNAP_COUNT" -eq 0 ] && [ "$TRADE_COUNT" -eq 0 ]; then
    echo "ERROR: No data found for ${REPORT_DATE}. Exiting."
    exit 1
fi

# Generate report
echo ""
echo "Generating daily report..."
python -m tools.daily_report_light \
    --data-root "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --symbol "${SYMBOL}" \
    --date "${REPORT_DATE}"

# List generated files
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}/"

# Upload report to GCS
echo ""
echo "Uploading report to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${GCS_BUCKET}/reports/${REPORT_DATE}/"

echo ""
echo "========================================"
echo "Report completed successfully!"
echo "Report available at: ${GCS_BUCKET}/reports/${REPORT_DATE}/"
echo "========================================"

# Cleanup
rm -rf "${WORK_DIR}"
