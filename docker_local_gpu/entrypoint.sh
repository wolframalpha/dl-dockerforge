#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — Starts SSHD + JupyterLab simultaneously
# (All CUDA extensions pre-compiled — no post-boot setup needed)
# =============================================================================
set -e

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    DL Toolkit (GPU Build) — CUDA 12.8 | All Compiled    ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Set root password (use ROOT_PASSWORD env var, default: 'dltoolkit') ───────
ROOT_PASSWORD="${ROOT_PASSWORD:-dltoolkit}"
echo "root:${ROOT_PASSWORD}" | chpasswd
echo -e "${GREEN}[✓]${NC} Root password set (change via ROOT_PASSWORD env var)"

# ── Start SSH daemon (on SSH_PORT to avoid conflict with host SSHD) ───────────
SSH_PORT="${SSH_PORT:-2222}"
/usr/sbin/sshd -p "${SSH_PORT}"
echo -e "${GREEN}[✓]${NC} SSH server started on port ${SSH_PORT}"
echo -e "    Connect via VS Code Remote-SSH:  ${YELLOW}ssh -p ${SSH_PORT} root@<host-ip>${NC}"

# ── Start JupyterLab ──────────────────────────────────────────────────────────
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_PASSWORD="${JUPYTER_PASSWORD:-dltoolkit}"

# Hash the password for Jupyter
JUPYTER_HASHED_PASSWORD=$(python -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))")

echo -e "${GREEN}[✓]${NC} Starting JupyterLab on port ${JUPYTER_PORT}"
echo -e "    URL: ${YELLOW}http://<host-ip>:${JUPYTER_PORT}/lab${NC}"
echo -e "    Password: (set via JUPYTER_PASSWORD env var)"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo -e "${CYAN}── GPU Status ──────────────────────────────────────────────${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
    echo ""
fi

# ── Library verification ─────────────────────────────────────────────────────
echo -e "${CYAN}── Compiled Libraries ──────────────────────────────────────${NC}"
python -c "import flash_attn; print(f'  Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "  Flash Attention: not available"
python -c "import bitsandbytes; print(f'  bitsandbytes:    {bitsandbytes.__version__}')" 2>/dev/null || echo "  bitsandbytes: not available"
python -c "import sglang; print(f'  SGLang:          {sglang.__version__}')" 2>/dev/null || echo "  SGLang: not available"
python -c "import deepspeed; print(f'  DeepSpeed:       {deepspeed.__version__}')" 2>/dev/null || echo "  DeepSpeed: not available"
echo -e "${GREEN}[✓]${NC} All CUDA extensions pre-compiled — no setup scripts needed!"
echo ""

# ── If extra args passed (e.g. 'bash'), run those instead of jupyter ──────────
if [ $# -gt 0 ]; then
    echo -e "${CYAN}── Running custom command ──────────────────────────────────${NC}"
    exec "$@"
else
    # Run JupyterLab in foreground (keeps container alive)
    exec jupyter lab \
        --ip=0.0.0.0 \
        --port="${JUPYTER_PORT}" \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.password="${JUPYTER_HASHED_PASSWORD}" \
        --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
        --notebook-dir=/workspace
fi
