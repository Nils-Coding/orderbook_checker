#!/usr/bin/env bash
# Robust sync of locally recorded Parquet data to GCS, with cleanup + alerting.
#
# Supersedes the older ~/sync_to_gcs_robust.sh. Keeps the good parts of it
# (timeouts, structured logging, GCS heartbeat) and ADDS the two things that
# were missing and caused the June outage:
#   1. RETENTION CLEANUP: after a successful upload, delete local Parquet files
#      older than RETENTION_DAYS (they are safely in the bucket). Without this
#      the disk fills up at ~3-5 GB/day and writes eventually fail (ENOSPC).
#   2. ntfy ALERT on sync failure, so a broken sync is noticed within the hour.
#
# rsync runs WITHOUT -d, so deleting local files never removes them from GCS.
# On failure NOTHING is deleted (local backlog is kept for manual re-upload).
#
# Run hourly via cron, e.g.:
#   0 * * * * /home/michaelscheland/orderbook_checker/scripts/sync_to_gcs.sh >> /home/michaelscheland/orderbook_checker/data/logs/sync_cron.log 2>&1

set -uo pipefail

BUCKET="${BUCKET:-gs://orderflow-data-lake/orderbook-checker}"
DATA_DIR="${DATA_DIR:-/home/michaelscheland/orderbook_checker/data}"
RETENTION_DAYS="${RETENTION_DAYS:-1}"
CONFIG_FILE="${CONFIG_FILE:-/home/michaelscheland/orderbook_checker/config.yaml}"

LOG_DIR="$DATA_DIR/logs"
SYNC_LOG="$LOG_DIR/sync.log"
HEARTBEAT_FILE="$BUCKET/heartbeat/orderbook-recorder.json"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$LOG_DIR" 2>/dev/null || true

log() { echo "$TIMESTAMP $1" >> "$SYNC_LOG"; }

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

fail() {
    local msg="$1"
    log "ERROR: $msg"
    notify "GCS sync FAILED" \
        "${msg}. Local data is KEPT (no cleanup) so it can be re-uploaded once fixed. Check VM network / gcloud auth / disk space." \
        "urgent" "rotating_light"
    exit 1
}

log "=========================================="
log "Starting sync"

# Ensure base directories exist. The recorder may not have created the trades
# dir yet (e.g. no trades recorded so far), and the cleanup below must never
# leave a base dir missing for the next run.
mkdir -p "$DATA_DIR/snapshots" "$DATA_DIR/trades" 2>/dev/null \
    || fail "Cannot create data directories under $DATA_DIR"

SNAP_BEFORE=$(find "$DATA_DIR/snapshots" -name "*.parquet" 2>/dev/null | wc -l)
TRADE_BEFORE=$(find "$DATA_DIR/trades" -name "*.parquet" 2>/dev/null | wc -l)
log "Local files before sync: snapshots=$SNAP_BEFORE, trades=$TRADE_BEFORE"

log "Syncing snapshots..."
timeout 600 gsutil -m rsync -r "$DATA_DIR/snapshots" "$BUCKET/snapshots/" >> "$SYNC_LOG" 2>&1 \
    || fail "Snapshot sync failed or timed out"
log "Snapshots synced"

log "Syncing trades..."
timeout 300 gsutil -m rsync -r "$DATA_DIR/trades" "$BUCKET/trades/" >> "$SYNC_LOG" 2>&1 \
    || fail "Trade sync failed or timed out"
log "Trades synced"

# Reports are non-critical (and may not exist).
if [ -d "$DATA_DIR/reports" ]; then
    log "Syncing reports..."
    timeout 60 gsutil -m rsync -r "$DATA_DIR/reports" "$BUCKET/reports/" >> "$SYNC_LOG" 2>&1 \
        || log "WARNING: Report sync failed (non-critical)"
fi

# --- Retention cleanup: only AFTER successful snapshot + trade upload ---
log "Cleanup: removing local Parquet files older than ${RETENTION_DAYS} day(s) (already uploaded)."
for sub in snapshots trades; do
    find "$DATA_DIR/$sub" -type f -name '*.parquet' -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
    # -mindepth 1 so the base dir ($DATA_DIR/$sub) itself is never removed,
    # only empty partition subdirectories below it.
    find "$DATA_DIR/$sub" -mindepth 1 -type d -empty -delete 2>/dev/null || true
done

# --- Heartbeat with metadata ---
SNAP_AFTER=$(find "$DATA_DIR/snapshots" -name "*.parquet" 2>/dev/null | wc -l)
TRADE_AFTER=$(find "$DATA_DIR/trades" -name "*.parquet" 2>/dev/null | wc -l)
DISK_USAGE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
DISK_AVAIL=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4}')
RECORDER_STATUS=$(systemctl is-active orderbook-recorder 2>/dev/null || echo "unknown")

HEARTBEAT_JSON=$(cat << EOF
{
  "timestamp": "$TIMESTAMP",
  "status": "ok",
  "hostname": "$(hostname)",
  "snapshots_count": $SNAP_AFTER,
  "trades_count": $TRADE_AFTER,
  "disk_usage": "$DISK_USAGE",
  "disk_available": "$DISK_AVAIL",
  "recorder_status": "$RECORDER_STATUS"
}
EOF
)

log "Writing heartbeat..."
echo "$HEARTBEAT_JSON" | gsutil cp - "$HEARTBEAT_FILE" 2>> "$SYNC_LOG" \
    || fail "Failed to write heartbeat"

gsutil cat "$HEARTBEAT_FILE" > /dev/null 2>&1 \
    || fail "Heartbeat validation failed (cannot read back)"

# Warn (don't fail) if the recorder is not active -- recording may be down even
# though the sync itself works.
if [ "$RECORDER_STATUS" != "active" ]; then
    notify "Recorder not active" \
        "The orderbook-recorder service is '${RECORDER_STATUS}' (not active). Recording may be stopped even though GCS sync works. Check the service." \
        "high" "warning"
fi

log "Sync completed successfully (snapshots=$SNAP_AFTER, trades=$TRADE_AFTER, free=$DISK_AVAIL)"
log "=========================================="
