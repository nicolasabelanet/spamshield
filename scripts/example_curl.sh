#!/usr/bin/env bash
# Usage:
#   API_KEY=your_key ./signed_request.sh http://localhost:8000/predict \
#     '{"texts":["Free cash now!"]}'

set -euo pipefail

API_URL="${1:-http://localhost:8000/predict}"
BODY="${2:-'{"texts":["hello"]}'}"

API_KEY="${API_KEY:-dev_key}"
API_SECRET="${API_SECRET:-$API_KEY}"   # if you later split key vs secret, set API_SECRET
TIMESTAMP=$(date +%s)

# Derive METHOD and PATH from URL (assumes no query string)
METHOD=POST
REQUEST_PATH="$(printf "%s" "$API_URL" | awk -F'//' '{print $2}' | cut -d/ -f2-)"
REQUEST_PATH="/${REQUEST_PATH#*/}"  # ensure leading slash

# Compute SHA256 of body (hex)
CONTENT_SHA256=$(printf "%s" "$BODY" | openssl dgst -sha256 -binary | xxd -p -c 256)

STRING_TO_SIGN="${METHOD}\n${REQUEST_PATH}\n${TIMESTAMP}\n${CONTENT_SHA256}\n${API_KEY}"

SIGNATURE=$(printf "%b" "$STRING_TO_SIGN" \
  | openssl dgst -sha256 -hmac "$API_SECRET" -binary | xxd -p -c 256)

curl -i -sS -X "$METHOD" "$API_URL" \
  -H "Content-Type: application/json; charset=utf-8" \
  -H "x-api-key: ${API_KEY}" \
  -H "x-timestamp: ${TIMESTAMP}" \
  -H "x-signature: ${SIGNATURE}" \
  -d "$BODY"
