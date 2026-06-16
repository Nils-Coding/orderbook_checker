#!/usr/bin/env bash
# Sync locally recorded Parquet data to GCS and clean up already-uploaded files.
#
# Strategy (prevents the "disk fills up -> writes fail -> recording stops" loop):
#   1. Mirror local data -> GCS bucket via `gsutil rsync` (WITHOUT -d, so a
#      local delete never removes anything from the bucket).
#   2. ONLY on a successful upload: delete local Parquet files older than
#      RETENTION_DAYS. They are safely in the bucket; the most recent data is
#      kept so the running recorder is never disturbed and a small window is
#      available for manual recovery.
#   3. On upload failure: delete NOTHING (keep the local backlog until the
#      problem is fixed -> manual re-upload possible), send an ntfy alert, and
#      exit non-zero.
#
# Intended to run hourly via cron. Example crontab line:
#   5 * * * * /home/michaelscheland/orderbook_checker/scripts/sync_to_gcs.sh >> /home/michaelscheland/orderbook_checker/data/logs/sync.log 2>&1

set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/home/michaelscheland/orderbook_checker/data}"
GCS_BUCKET="${GCS_BUCKET:-gs://orderflow-data-lake/orderbook-checker}"
RETENTION_DAYS="${RETENTION_DAYS:-1}"
CONFIG_FILE="${CONFIG_FILE:-/home/michaelscheland/orderbook_checker/config.yaml}"

# --- ntfy helper (reads topic/host from config.yaml, overridable via env) ---
read_cfg() { grep -E "^$1:" "$CONFIG_FILE" 2>/dev/null | head -1 | sed -E 's/^[^:]+:[[:space:]]*"?([^"#]*)"?.*/\1/' | xargs; }
NTFY_TOPIC="${NTFY_TOPIC:-$(read_cfg notify_topic)}"
HOST_NAME="${HOST_NAME:-$(read_cfg host_name)}"
HOST_NAME="${HOST_NAME:-$(hostname)}"

notify() {
    local title="$1" message="$2" priority="${3:-default}" tags="${4:-}"
    [ -z "$NTFY_TOPIC" ] && return 0
    curl -fsS \
        -H "Title: [${HOST_NAME}] ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${message}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

echo "[$(date -u +%FT%TZ)] Starting GCS sync from ${DATA_ROOT} -> ${GCS_BUCKET}"

ok=true
for sub in snapshots trades; do
    if [ -d "${DATA_ROOT}/${sub}" ]; then
        gsutil -m rsync -r "${DATA_ROOT}/${sub}" "${GCS_BUCKET}/${sub}" || ok=false
    fi
done

if [ "$ok" != true ]; then
    echo "ERROR: gsutil rsync failed; keeping all local files, no cleanup."
    notify "GCS sync FAILED" \
        "gsutil rsync to ${GCS_BUCKET} failed. Local data is KEPT (no cleanup) so it can be re-uploaded once fixed. Check VM network / gcloud auth / disk space." \
        "urgent" "rotating_light"
    exit 1
fi

echo "Sync OK. Removing local Parquet files older than ${RETENTION_DAYS} day(s) (already uploaded)."
for sub in snapshots trades; do
    [ -d "${DATA_ROOT}/${sub}" ] || continue
    find "${DATA_ROOT}/${sub}" -type f -name '*.parquet' -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null
    # Remove now-empty hour/date directories left behind.
    find "${DATA_ROOT}/${sub}" -type d -empty -delete 2>/dev/null
done

echo "[$(date -u +%FT%TZ)] GCS sync + cleanup complete."
