#!/usr/bin/env -S bash
# NOTE: Save this file with LF line endings. If you see 'bash\r' errors run:
#       sed -i 's/\r$//' install-sdrwatch.sh
# install-sdrwatch.sh — One‑shot installer for SDRwatch (Pi 4/5, Raspberry Pi OS 64‑bit / Debian 13 "Trixie")
#
# What this does
#  1) Installs system deps for RTL‑SDR, HackRF, SoapySDR, NumPy/SciPy, Flask, etc. 
#  2) Creates a Python venv that can see APT packages via --system-site-packages
#  3) Pip‑installs lightweight Python deps (Flask, pyrtlsdr, rich)
#  4) Verifies rtl_test / hackrf_info; applies udev rules + kernel blacklist for RTL2832U
#  5) Uses /var/lib,/var/cache,/run for state to play nice with ProtectHome/ProtectSystem
#  6) (Optional) Installs systemd services hardened with StateDirectory/RuntimeDirectory
#
# Safe to re‑run; idempotent where possible. Non‑interactive: set SDRWATCH_AUTO_YES=1

set -Eeuo pipefail

# -----------------------------
# Config (override via env before running)
# -----------------------------
PROJECT_DIR=${PROJECT_DIR:-"$PWD"}
# Normalize to absolute path for systemd unit directives
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python3"
PIP_BIN="${VENV_DIR}/bin/pip"

WEB_DASH="${PROJECT_DIR}/sdrwatch-web-simple.py"      # dash
WEB_UNDERSCORE="${PROJECT_DIR}/sdrwatch_web_simple.py" # underscore (services default)
CONTROL_PY="${PROJECT_DIR}/sdrwatch-control.py"
CORE_PY="${PROJECT_DIR}/sdrwatch.py"

# Put state under /var to avoid home permission issues with ProtectHome.
STATE_DIR_DEFAULT=${STATE_DIR_DEFAULT:-"/var/lib/sdrwatch"}
CACHE_DIR_DEFAULT=${CACHE_DIR_DEFAULT:-"/var/cache/sdrwatch"}
RUNTIME_DIR_DEFAULT=${RUNTIME_DIR_DEFAULT:-"/run/sdrwatch"}
CONTROL_RUNTIME_DIR_DEFAULT=${CONTROL_RUNTIME_DIR_DEFAULT:-"/run/sdrwatch-control"}
DB_PATH_DEFAULT=${DB_PATH_DEFAULT:-"${STATE_DIR_DEFAULT}/sdrwatch.db"}

# systemd defaults (editable during interactive step)
SRV_USER_DEFAULT=${SRV_USER_DEFAULT:-"${SUDO_USER:-$USER}"}
SRV_GROUP_DEFAULT=${SRV_GROUP_DEFAULT:-"${SRV_USER_DEFAULT}"}
SDRWATCH_CONTROL_HOST_DEFAULT=${SDRWATCH_CONTROL_HOST_DEFAULT:-"127.0.0.1"}
SDRWATCH_CONTROL_PORT_DEFAULT=${SDRWATCH_CONTROL_PORT_DEFAULT:-"8765"}
SDRWATCH_WEB_HOST_DEFAULT=${SDRWATCH_WEB_HOST_DEFAULT:-"0.0.0.0"}
SDRWATCH_WEB_PORT_DEFAULT=${SDRWATCH_WEB_PORT_DEFAULT:-"8080"}
SDRWATCH_CONTROL_TOKEN_DEFAULT=${SDRWATCH_CONTROL_TOKEN_DEFAULT:-"change_me_control"}
SDRWATCH_TOKEN_DEFAULT=${SDRWATCH_TOKEN_DEFAULT:-""}
ENV_FILE_DEFAULT=${ENV_FILE_DEFAULT:-"/etc/sdrwatch.env"}
UNIT_CTL_DEFAULT=${UNIT_CTL_DEFAULT:-"/etc/systemd/system/sdrwatch-control.service"}
UNIT_WEB_DEFAULT=${UNIT_WEB_DEFAULT:-"/etc/systemd/system/sdrwatch-web.service"}
# Where the service code will be deployed to (outside /home to work with ProtectHome)
DEPLOY_DIR_DEFAULT=${DEPLOY_DIR_DEFAULT:-"/opt/sdrwatch"}

# -----------------------------
# Helpers
# -----------------------------
log(){ printf "[install] %s\n" "$*"; }
die(){ printf "[install:ERROR] %s\n" "$*" >&2; exit 1; }
require(){ command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"; }

prompt_default(){
  local prompt="$1"; local def="$2"; local var
  if [[ "${SDRWATCH_AUTO_YES:-}" == 1 ]]; then echo "$def"; return 0; fi
  read -rp "$prompt [$def]: " var || true
  echo "${var:-$def}"
}

prompt_yn(){
  local prompt="$1"; local def="${2:-y}"; local ans
  if [[ "${SDRWATCH_AUTO_YES:-}" == 1 ]]; then echo y; return 0; fi
  while true; do
    read -rp "$prompt (${def^^}/$([[ $def == y ]] && echo N || echo Y)): " ans || ans="$def"
    ans=${ans:-$def}; ans=${ans,,}
    case "$ans" in y|yes) echo y; return 0;; n|no) echo n; return 0;; esac
  done
}

# -----------------------------
# Pre-flight
# -----------------------------
require sudo
require python3

# Detect SoapySDR core library package (versioned name varies by repo)
SOAPY_CORE_PKG=""
for candidate in libsoapysdr0.10 libsoapysdr0.9 libsoapysdr; do
  if apt-cache show "$candidate" >/dev/null 2>&1; then
    SOAPY_CORE_PKG="$candidate"
    break
  fi
done
SOAPY_DEV_PKG=""
if apt-cache show libsoapysdr-dev >/dev/null 2>&1; then
  SOAPY_DEV_PKG="libsoapysdr-dev"
fi
if [[ -z "$SOAPY_CORE_PKG" ]]; then
  log "WARNING: No versioned libsoapysdr package found (will rely on python3-soapysdr pulling it or existing install)."
else
  log "Detected SoapySDR core package: $SOAPY_CORE_PKG"
fi

log "Updating APT and installing system packages (Trixie)…"
sudo apt update
sudo apt install -y \
  git curl ca-certificates build-essential cmake pkg-config \
  libusb-1.0-0 libusb-1.0-0-dev \
  python3-venv python3-dev \
  python3-numpy python3-scipy \
  python3-soapysdr $SOAPY_CORE_PKG $SOAPY_DEV_PKG \
  librtlsdr0 librtlsdr-dev rtl-sdr \
  soapysdr-module-rtlsdr soapysdr-module-hackrf \
  soapysdr-tools \
  hackrf || die "APT install failed (see above)."

# -----------------------------
# Verify / (optional) build rtl-sdr if broken
# -----------------------------
if ! rtl_test -t >/dev/null 2>&1; then
  log "rtl_test not working — building rtl-sdr from source…"
  WORKROOT="${PROJECT_DIR}/.build-rtl-sdr"; rm -rf "$WORKROOT"; mkdir -p "$WORKROOT"; pushd "$WORKROOT" >/dev/null
  SRC_DIR=""
  if git clone --depth=1 https://github.com/rtlsdrblog/rtl-sdr-blog.git; then
    SRC_DIR="rtl-sdr-blog"
  else
    git clone --depth=1 https://github.com/osmocom/rtl-sdr.git || die "Failed to clone rtl-sdr sources"
    SRC_DIR="rtl-sdr"
  fi
  cd "$SRC_DIR"; mkdir -p build; cd build
  cmake -DDETACH_KERNEL_DRIVER=ON -DCPACK_PACKAGING_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_PREFIX=/usr ..
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  if [ -f ../rtl-sdr.rules ]; then
    sudo cp -v ../rtl-sdr.rules /etc/udev/rules.d/rtl-sdr.rules
    sudo udevadm control --reload-rules || true
    sudo udevadm trigger || true
  fi
  popd >/dev/null
else
  log "rtl_test looks OK; skipping source build."
fi

# -----------------------------
# Kernel module blacklist (prevents DVB from grabbing RTL2832U)
# -----------------------------
BLACKLIST="/etc/modprobe.d/rtl-sdr-blacklist.conf"
if [ ! -f "$BLACKLIST" ] || ! grep -q "dvb_usb_rtl28xxu" "$BLACKLIST" 2>/dev/null; then
  log "Writing kernel blacklist at $BLACKLIST"
  sudo bash -c "cat > '$BLACKLIST'" <<'EOF'
# Prevent the DVB drivers from grabbing RTL2832U-based dongles.
blacklist dvb_usb_rtl28xxu
blacklist rtl2832
blacklist rtl2830
EOF
fi

# -----------------------------
# Prepare state/cache/runtime directories
# -----------------------------
for d in "$STATE_DIR_DEFAULT" "$CACHE_DIR_DEFAULT" "$RUNTIME_DIR_DEFAULT" "$CONTROL_RUNTIME_DIR_DEFAULT"; do
  sudo install -d -m 0755 -o root -g root "$d"
done

# -----------------------------
# Python venv with system site packages
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
  log "Creating venv at $VENV_DIR (with --system-site-packages)"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip tooling…"
$PIP_BIN install -U pip setuptools wheel

# Keep pip light; heavy numerics come from APT
REQS_FILE="$PROJECT_DIR/requirements.sdrwatch.txt"
cat > "$REQS_FILE" <<'REQS'
# Light Python deps; NumPy/SciPy/SoapySDR come from APT via system site packages
flask>=3.0.0
pyrtlsdr
rich>=13.0.0
REQS

log "Installing Python packages from $REQS_FILE"
$PIP_BIN install -r "$REQS_FILE"

# -----------------------------
# Filename compatibility (underscore vs dash)
# -----------------------------
if [ -f "$WEB_DASH" ] && [ ! -e "$WEB_UNDERSCORE" ]; then
  log "Creating compatibility symlink: $(basename "$WEB_UNDERSCORE") → $(basename "$WEB_DASH")"
  ln -s "$(basename "$WEB_DASH")" "$WEB_UNDERSCORE"
fi

# -----------------------------
# Sanity checks
# -----------------------------
log "Python import checks:"
$PYTHON_BIN - <<'PY'
try:
    import numpy, scipy, SoapySDR
    print('[check] numpy/scipy/SoapySDR import: OK')
except Exception as e:
    print(f'[check] numpy/scipy/SoapySDR: {e}')
try:
    import flask
    print('[check] Flask import: OK')
except Exception as e:
    print(f'[check] Flask: {e}')
try:
    from rtlsdr import RtlSdr
    print('[check] pyrtlsdr import: OK')
except Exception as e:
    print(f'[check] pyrtlsdr: {e}')
PY

log "Verifying SDR CLIs…"
if rtl_test -t >/dev/null 2>&1; then log "rtl_test: OK"; else log "rtl_test: NOT OK (replug/reboot may be required)"; fi
if hackrf_info >/dev/null 2>&1; then log "hackrf_info: OK"; else log "hackrf_info: NOT FOUND/ERROR (only needed for HackRF)"; fi

# =============================================================
# Interactive systemd services setup (Recommended)
# =============================================================
if [ "$(prompt_yn 'Install and start SDRwatch services now? (Recommended)' y)" = y ]; then
  log "Collecting service settings…"
  SRV_USER=$(prompt_default "Service user" "$SRV_USER_DEFAULT")
  SRV_GROUP=$(prompt_default "Service group" "$SRV_GROUP_DEFAULT")
  DB_PATH=$(prompt_default "Database path" "$DB_PATH_DEFAULT")
  DEPLOY_DIR=$(prompt_default "Deployment directory for service code" "$DEPLOY_DIR_DEFAULT")
  CONTROL_HOST=$(prompt_default "Controller host" "$SDRWATCH_CONTROL_HOST_DEFAULT")
  CONTROL_PORT=$(prompt_default "Controller port" "$SDRWATCH_CONTROL_PORT_DEFAULT")
  WEB_HOST=$(prompt_default "Web host" "$SDRWATCH_WEB_HOST_DEFAULT")
  WEB_PORT=$(prompt_default "Web port" "$SDRWATCH_WEB_PORT_DEFAULT")
  CONTROL_TOKEN=$(prompt_default "Controller API token" "$SDRWATCH_CONTROL_TOKEN_DEFAULT")
  WEB_TOKEN=$(prompt_default "Web page/API token (optional)" "$SDRWATCH_TOKEN_DEFAULT")
  ENV_FILE=$(prompt_default "Env file location" "$ENV_FILE_DEFAULT")
  UNIT_CTL=$(prompt_default "Controller unit path" "$UNIT_CTL_DEFAULT")
  UNIT_WEB=$(prompt_default "Web unit path" "$UNIT_WEB_DEFAULT")

  # -----------------------------
  # Deploy service code to /opt (or chosen dir) to avoid ProtectHome issues
  # -----------------------------
  log "Deploying SDRwatch code to $DEPLOY_DIR"
  sudo install -d -m 0755 -o root -g root "$DEPLOY_DIR"
  # Sync code (exclude venvs, git, build artifacts)
  if command -v rsync >/dev/null 2>&1; then
    sudo rsync -a --delete \
      --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
      --exclude '.build-rtl-sdr/' --exclude '*.pyc' \
      "$PROJECT_DIR/" "$DEPLOY_DIR/"
  else
    log "rsync not found; using tar to copy files"
    tmp_tar="$(mktemp /tmp/sdrwatch-copy.XXXXXX.tar)"
    (cd "$PROJECT_DIR" && tar --exclude .git --exclude .venv --exclude __pycache__ --exclude .build-rtl-sdr -cf "$tmp_tar" .)
    sudo tar -xf "$tmp_tar" -C "$DEPLOY_DIR"
    rm -f "$tmp_tar"
  fi

  # Service venv lives with the deployed code
  DEPLOY_VENV_DIR="$DEPLOY_DIR/.venv"
  DEPLOY_PY="$DEPLOY_VENV_DIR/bin/python3"
  DEPLOY_PIP="$DEPLOY_VENV_DIR/bin/pip"
  if [ ! -d "$DEPLOY_VENV_DIR" ]; then
    log "Creating service venv at $DEPLOY_VENV_DIR (with --system-site-packages)"
    sudo python3 -m venv --system-site-packages "$DEPLOY_VENV_DIR"
  fi
  # Install Python deps inside deployed venv
  log "Installing Python packages into service venv"
  sudo "$DEPLOY_PIP" install -U pip setuptools wheel
  # Reuse same lightweight requirements
  sudo "$DEPLOY_PIP" install -r "$DEPLOY_DIR/requirements.sdrwatch.txt"

  log "Writing env file to $ENV_FILE"
  sudo install -m 0640 -o root -g "$SRV_GROUP" /dev/null "$ENV_FILE"
  sudo bash -c "cat > '$ENV_FILE'" <<EOF
# Auto-generated by install-sdrwatch.sh (Trixie)
SDRWATCH_PROJECT_DIR="$DEPLOY_DIR"
SDRWATCH_VENV_BIN="$DEPLOY_VENV_DIR/bin"
SDRWATCH_DB="$DB_PATH"
SDRWATCH_STATE_DIR="$STATE_DIR_DEFAULT"
SDRWATCH_CACHE_DIR="$CACHE_DIR_DEFAULT"
SDRWATCH_RUNTIME_DIR="$RUNTIME_DIR_DEFAULT"
SDRWATCH_CONTROL_RUNTIME_DIR="$CONTROL_RUNTIME_DIR_DEFAULT"
SDRWATCH_CONTROL_HOST="$CONTROL_HOST"
SDRWATCH_CONTROL_PORT="$CONTROL_PORT"
SDRWATCH_CONTROL_TOKEN="$CONTROL_TOKEN"
SDRWATCH_WEB_HOST="$WEB_HOST"
SDRWATCH_WEB_PORT="$WEB_PORT"
SDRWATCH_TOKEN="$WEB_TOKEN"
# steer scratch off /tmp (tmpfs on Trixie)
TMPDIR="/var/tmp/sdrwatch"
EOF

  sudo install -d -m 0770 -o "$SRV_USER" -g "$SRV_GROUP" /var/tmp/sdrwatch "$STATE_DIR_DEFAULT" "$CACHE_DIR_DEFAULT"

  log "Writing systemd unit: $UNIT_CTL"
  sudo bash -c "cat > '$UNIT_CTL'" <<'UNIT'
[Unit]
Description=SDRwatch Control API (manager for sdrwatch.py jobs)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=SRV_USER
Group=SRV_GROUP
EnvironmentFile=/etc/sdrwatch.env
WorkingDirectory=${SDRWATCH_PROJECT_DIR}
# Ensure state/cache/runtime exist and are writable
StateDirectory=sdrwatch
RuntimeDirectory=sdrwatch-control
CacheDirectory=sdrwatch
# Allow writing to our state & runtime directories
ReadWritePaths=/var/lib/sdrwatch /run/sdrwatch-control /var/cache/sdrwatch /var/tmp/sdrwatch ${SDRWATCH_PROJECT_DIR}
ExecStart=/bin/bash -lc 'cd "$SDRWATCH_PROJECT_DIR" && exec "$SDRWATCH_VENV_BIN"/python3 sdrwatch-control.py serve --host "$SDRWATCH_CONTROL_HOST" --port "$SDRWATCH_CONTROL_PORT" --token "$SDRWATCH_CONTROL_TOKEN"'
Restart=on-failure
RestartSec=2
StandardOutput=journal
StandardError=journal

NoNewPrivileges=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
UNIT
  # Substitute runtime values (make absolute paths literal)
  sudo sed -i "s/SRV_USER/$SRV_USER/g; s/SRV_GROUP/$SRV_GROUP/g; s|/etc/sdrwatch.env|$ENV_FILE|g; s|\${SDRWATCH_PROJECT_DIR}|$DEPLOY_DIR|g" "$UNIT_CTL"

  # Web unit
  log "Writing systemd unit: $UNIT_WEB"
  sudo bash -c "cat > '$UNIT_WEB'" <<'UNIT'
[Unit]
Description=SDRwatch Web (simple) — Flask UI
After=network-online.target sdrwatch-control.service
Wants=network-online.target sdrwatch-control.service

[Service]
Type=simple
User=SRV_USER
Group=SRV_GROUP
EnvironmentFile=/etc/sdrwatch.env
WorkingDirectory=${SDRWATCH_PROJECT_DIR}
# Directories
StateDirectory=sdrwatch
RuntimeDirectory=sdrwatch
CacheDirectory=sdrwatch
ReadWritePaths=/var/lib/sdrwatch /run/sdrwatch /var/cache/sdrwatch /var/tmp/sdrwatch ${SDRWATCH_PROJECT_DIR}
ExecStart=/bin/bash -lc 'cd "$SDRWATCH_PROJECT_DIR" && \
  SDRWATCH_CONTROL_URL="http://$SDRWATCH_CONTROL_HOST:$SDRWATCH_CONTROL_PORT" \
  SDRWATCH_CONTROL_TOKEN="$SDRWATCH_CONTROL_TOKEN" \
  SDRWATCH_TOKEN="$SDRWATCH_TOKEN" \
  exec "$SDRWATCH_VENV_BIN"/python3 sdrwatch_web_simple.py --db "$SDRWATCH_DB" --host "$SDRWATCH_WEB_HOST" --port "$SDRWATCH_WEB_PORT"'
Restart=on-failure
RestartSec=2
StandardOutput=journal
StandardError=journal

NoNewPrivileges=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
UNIT
  sudo sed -i "s/SRV_USER/$SRV_USER/g; s/SRV_GROUP/$SRV_GROUP/g; s|/etc/sdrwatch.env|$ENV_FILE|g; s|\${SDRWATCH_PROJECT_DIR}|$DEPLOY_DIR|g" "$UNIT_WEB"

  log "Reloading systemd and enabling services…"
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$(basename "$UNIT_CTL")"
  sleep 1
  sudo systemctl enable --now "$(basename "$UNIT_WEB")"

  log "Services installed."
else
  log "Skipping systemd services per user choice. You can re-run this installer later."
fi

# -----------------------------
# Summary / next steps
# -----------------------------
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || true
cat <<EOS

[done] SDRwatch install complete (Trixie-compatible).

Next steps:
  • Reboot recommended so udev + blacklist take effect:   sudo reboot
  • Activate venv in your shell:                          source "$VENV_DIR/bin/activate"
  • Quick scan sanity (FM band):
      $PYTHON_BIN "$CORE_PY" --driver rtlsdr --start 88e6 --stop 108e6 --fft 4096 --avg 8 --db "$DB_PATH_DEFAULT"

If you enabled services:
  • Control API:  http://${SDRWATCH_CONTROL_HOST_DEFAULT}:${SDRWATCH_CONTROL_PORT_DEFAULT}
  • Web UI:       http://${HOST_IP:-<your-pi-ip>}:${SDRWATCH_WEB_PORT_DEFAULT}
  • Logs:         sudo journalctl -u sdrwatch-control -f
                  sudo journalctl -u sdrwatch-web -f

Non‑interactive mode:
  SDRWATCH_AUTO_YES=1 ./install-sdrwatch.sh

EOS
