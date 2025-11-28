#!/usr/bin/env -S bash
# uninstall-sdrwatch.sh — Stop/remove SDRwatch services and clean installer artifacts.
# Leaves your database (SDRWATCH_DB) intact but removes unit files, env files,
# deployment directories, virtualenvs, runtime dirs, etc. Run from the repo root.

set -Eeuo pipefail

log(){ printf "[uninstall] %s\n" "$*"; }
die(){ printf "[uninstall:ERROR] %s\n" "$*" >&2; exit 1; }
require(){ command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"; }

require sudo
require systemctl

PROJECT_DIR=${PROJECT_DIR:-"$PWD"}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
VENV_DIR="$PROJECT_DIR/.venv"
WEB_UNDERSCORE="$PROJECT_DIR/sdrwatch_web_simple.py"

ENV_FILE=${ENV_FILE:-"/etc/sdrwatch.env"}
UNIT_CTL=${UNIT_CTL:-"/etc/systemd/system/sdrwatch-control.service"}
UNIT_WEB=${UNIT_WEB:-"/etc/systemd/system/sdrwatch-web.service"}
DEPLOY_DIR_DEFAULT=${DEPLOY_DIR_DEFAULT:-"/opt/sdrwatch"}
STATE_DIR_DEFAULT=${STATE_DIR_DEFAULT:-"/var/lib/sdrwatch"}
CACHE_DIR_DEFAULT=${CACHE_DIR_DEFAULT:-"/var/cache/sdrwatch"}
RUNTIME_DIR_DEFAULT=${RUNTIME_DIR_DEFAULT:-"/run/sdrwatch"}
CONTROL_RUNTIME_DIR_DEFAULT=${CONTROL_RUNTIME_DIR_DEFAULT:-"/run/sdrwatch-control"}
TMP_DIR_DEFAULT=${TMP_DIR_DEFAULT:-"/var/tmp/sdrwatch"}

DEPLOY_DIR="$DEPLOY_DIR_DEFAULT"
STATE_DIR="$STATE_DIR_DEFAULT"
CACHE_DIR="$CACHE_DIR_DEFAULT"
RUNTIME_DIR="$RUNTIME_DIR_DEFAULT"
CONTROL_RUNTIME_DIR="$CONTROL_RUNTIME_DIR_DEFAULT"
TMP_DIR="$TMP_DIR_DEFAULT"
SERVICE_VENV_DIR=""
DB_PATH=""

if [[ -f "$ENV_FILE" ]]; then
  log "Loading installer env file: $ENV_FILE"
  # shellcheck source=/dev/null
  set -a; source "$ENV_FILE"; set +a || true
  DEPLOY_DIR="${SDRWATCH_PROJECT_DIR:-$DEPLOY_DIR}"
  STATE_DIR="${SDRWATCH_STATE_DIR:-$STATE_DIR}"
  CACHE_DIR="${SDRWATCH_CACHE_DIR:-$CACHE_DIR}"
  RUNTIME_DIR="${SDRWATCH_RUNTIME_DIR:-$RUNTIME_DIR}"
  CONTROL_RUNTIME_DIR="${SDRWATCH_CONTROL_RUNTIME_DIR:-$CONTROL_RUNTIME_DIR}"
  TMP_DIR="${TMPDIR:-$TMP_DIR}"
  DB_PATH="${SDRWATCH_DB:-}"  # informational only
  if [[ -n "${SDRWATCH_VENV_BIN:-}" ]]; then
    SERVICE_VENV_DIR="$(dirname "${SDRWATCH_VENV_BIN}")/.."
    SERVICE_VENV_DIR="$(cd "$SERVICE_VENV_DIR" 2>/dev/null && pwd -P)"
  fi
fi

stop_disable_unit(){
  local unit="$1"
  if systemctl list-unit-files "$unit" >/dev/null 2>&1; then
    log "Stopping $unit"
    sudo systemctl stop "$unit" 2>/dev/null || true
    log "Disabling $unit"
    sudo systemctl disable "$unit" 2>/dev/null || true
  fi
}

remove_unit_file(){
  local path="$1"
  if [[ -f "$path" ]]; then
    log "Removing unit file $path"
    sudo rm -f "$path"
  fi
}

safe_remove_dir(){
  local path="$1"
  local desc="$2"
  if [[ -z "$path" || "$path" == "/" ]]; then
    log "Skipping removal of $desc (invalid path)"
    return
  fi
  if [[ ! -d "$path" ]]; then
    return
  fi
  log "Removing $desc at $path"
  sudo rm -rf "$path"
}

log "Stopping and disabling systemd units (if present)…"
stop_disable_unit "sdrwatch-control.service"
stop_disable_unit "sdrwatch-web.service"
# Also stop custom paths if provided
if [[ -f "$UNIT_CTL" ]]; then stop_disable_unit "$(basename "$UNIT_CTL")"; fi
if [[ -f "$UNIT_WEB" ]]; then stop_disable_unit "$(basename "$UNIT_WEB")"; fi

log "Removing systemd unit files"
remove_unit_file "$UNIT_CTL"
remove_unit_file "$UNIT_WEB"

if [[ -f "$ENV_FILE" ]]; then
  log "Removing env file $ENV_FILE"
  sudo rm -f "$ENV_FILE"
fi

if [[ -n "$SERVICE_VENV_DIR" && -d "$SERVICE_VENV_DIR" ]]; then
  safe_remove_dir "$SERVICE_VENV_DIR" "service virtualenv"
fi

if [[ -n "$DEPLOY_DIR" && -d "$DEPLOY_DIR" ]]; then
  safe_remove_dir "$DEPLOY_DIR" "deployed SDRwatch code"
fi

safe_remove_dir "$RUNTIME_DIR" "runtime directory"
safe_remove_dir "$CONTROL_RUNTIME_DIR" "control runtime directory"
safe_remove_dir "$CACHE_DIR" "cache directory"
safe_remove_dir "$TMP_DIR" "temporary directory"

if [[ -d "$STATE_DIR" ]]; then
  log "Leaving state directory $STATE_DIR (contains DB/media). Remove manually if desired."
fi

if [[ -d "$VENV_DIR" ]]; then
  log "Removing project venv at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ -L "$WEB_UNDERSCORE" ]]; then
  log "Removing compatibility symlink $WEB_UNDERSCORE"
  rm -f "$WEB_UNDERSCORE"
fi

log "Reloading systemd daemon"
sudo systemctl daemon-reload

log "Uninstall complete."
if [[ -n "$DB_PATH" ]]; then
  log "Database preserved at: $DB_PATH"
else
  log "Database (if any) in $STATE_DIR was left untouched."
fi
printf '\nYou can now rerun install-sdrwatch.sh for a fresh deployment.\n'
