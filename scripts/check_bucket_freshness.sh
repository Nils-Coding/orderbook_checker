#!/usr/bin/env bash
# Dead-man's switch: alert via ntfy if no fresh data has landed in the GCS
# bucket recently. This watches the actual deliverable (new objects in the
# bucket), so it catches ANY failure mode -- recorder dead, writer stalled,
# sync job broken, VM down -- not just the ones we anticipated in code.
#
# IMPORTANT: data only lands in the bucket when sync_to_gcs.sh runs (hourly),
# so MAX_AGE_MIN must be larger than the sync interval (default 120 min for an
# hourly sync). Run this every 15-30 min via cron. Example crontab line:
#   */20 * * * * /home/michaelscheland/orderbook_checker/scripts/check_bucket_freshness.sh >> /home/michaelscheland/orderbook_checker/data/logs/watchdog.log 2>&1

set -uo pipefail

GCS_BUCKET="${GCS_BUCKET:-gs://orderflow-data-lake/orderbook-checker}"
SYMBOL="${SYMBOL:-BTCUSDT}"
MAX_AGE_MIN="${MAX_AGE_MIN:-120}"
CONFIG_FILE="${CONFIG_FILE:-/home/michaelscheland/orderbook_checker/config.yaml}"

read_cfg() { grep -E "^$1:" "$CONFIG_FILE" 2>/dev/null | head -1 | sed -E 's/^[^:]+:[[:space:]]*"?([^"#]*)"?.*/\1/' | xargs; }
NTFY_TOPIC="${NTFY_TOPIC:-$(read_cfg notify_topic)}"
HOST_NAME="${HOST_NAME:-$(read_cfg host_name)}"
HOST_NAME="${HOST_NAME:-watchdog}"

notify() {
    local title="$1" message="$2" priority="${3:-high}" tags="${4:-warning}"
    [ -z "$NTFY_TOPIC" ] && return 0
    curl -fsS \
        -H "Title: [${HOST_NAME}] ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${message}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

base="${GCS_BUCKET}/snapshots/symbol=${SYMBOL}/"

# Newest date partition under the symbol prefix.
latest_date_prefix=$(gsutil ls "$base" 2>/dev/null | sort | tail -1)
if [ -z "$latest_date_prefix" ]; then
    notify "Bucket watchdog: NO DATA" \
        "No snapshot partitions found under ${base}. Recording/sync may be completely down." \
        "urgent" "rotating_light"
    exit 1
fi

# Newest object's ISO8601 timestamp under that partition (gsutil ls -l puts the
# creation time in column 2 for object lines).
newest_ts=$(gsutil ls -l "${latest_date_prefix}**" 2>/dev/null \
    | awk '{print $2}' \
    | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' \
    | sort | tail -1)

if [ -z "$newest_ts" ]; then
    notify "Bucket watchdog: NO OBJECTS" \
        "Latest partition ${latest_date_prefix} contains no objects. New data is not arriving." \
        "urgent" "rotating_light"
    exit 1
fi

newest_epoch=$(date -d "$newest_ts" +%s 2>/dev/null || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$newest_ts" +%s 2>/dev/null)
now_epoch=$(date -u +%s)
age_min=$(( (now_epoch - newest_epoch) / 60 ))

if [ "$age_min" -gt "$MAX_AGE_MIN" ]; then
    notify "Bucket watchdog: STALE data" \
        "Newest object in ${base} is ${age_min} min old (threshold ${MAX_AGE_MIN} min). New data is NOT arriving in GCS -- the recorder or the sync job has likely stopped. Check the VM now." \
        "urgent" "rotating_light"
    exit 1
fi

echo "[$(date -u +%FT%TZ)] OK: newest bucket object is ${age_min} min old (threshold ${MAX_AGE_MIN})."
