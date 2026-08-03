#!/usr/bin/env bash
set -euo pipefail

: "${DEPLOY_HOST:?DEPLOY_HOST is required}"
: "${DEPLOY_USER:?DEPLOY_USER is required}"
: "${DEPLOY_PATH:?DEPLOY_PATH is required}"

DEPLOY_PORT="${DEPLOY_PORT:-22}"
DEPLOY_SHA="${GITHUB_SHA:-manual}"

[[ "$DEPLOY_HOST" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid DEPLOY_HOST" >&2; exit 2; }
[[ "$DEPLOY_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid DEPLOY_USER" >&2; exit 2; }
[[ "$DEPLOY_PORT" =~ ^[0-9]{1,5}$ ]] || { echo "Invalid DEPLOY_PORT" >&2; exit 2; }
[[ "$DEPLOY_PATH" =~ ^/[A-Za-z0-9._/-]+$ && "$DEPLOY_PATH" != *".."* ]] || {
  echo "DEPLOY_PATH must be a safe absolute path" >&2
  exit 2
}
[[ "$DEPLOY_SHA" =~ ^([a-f0-9]{7,64}|manual)$ ]] || { echo "Invalid deployment SHA" >&2; exit 2; }

archive="$(mktemp "${RUNNER_TEMP:-/tmp}/opcenter-deploy.XXXXXX.tgz")"
trap 'rm -f "$archive"' EXIT
remote_archive="/tmp/opcenter-${DEPLOY_SHA:0:12}.tgz"
target="${DEPLOY_USER}@${DEPLOY_HOST}"

tar \
  --exclude='./.git' \
  --exclude='./.env' \
  --exclude='./.env.*' \
  --exclude='./manuals' \
  --exclude='./indexes' \
  --exclude='./data' \
  --exclude='./evaluation_results' \
  --exclude='./production_monitor_report.json' \
  --exclude='./production_issue.md' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "$archive" .

scp -P "$DEPLOY_PORT" "$archive" "${target}:${remote_archive}"
ssh -p "$DEPLOY_PORT" "$target" \
  "set -e; test -f '${DEPLOY_PATH}/.env'; mkdir -p '${DEPLOY_PATH}'; tar -xzf '${remote_archive}' -C '${DEPLOY_PATH}'; rm -f '${remote_archive}'; cd '${DEPLOY_PATH}'; docker compose up -d --build --wait; docker compose ps"
