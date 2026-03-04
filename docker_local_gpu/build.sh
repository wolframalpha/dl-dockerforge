#!/usr/bin/env bash
# =============================================================================
# build.sh — Build & export the DL toolkit image using Docker (Windows/WSL2)
# Usage:  ./build.sh [build|export|all]
# =============================================================================
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-dl-toolkit-gpu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TAR_FILE="${TAR_FILE:-dl-toolkit-gpu.tar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*" >&2; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }

# ── Check Docker is available ─────────────────────────────────────────────────
check_docker() {
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed."
        err "On Windows: Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
        err "On Linux:   sudo apt install docker.io  (or install Docker Engine)"
        exit 1
    fi
    # Ensure Docker daemon is running
    if ! docker info &>/dev/null 2>&1; then
        err "Docker daemon is not running. Please start Docker Desktop or the Docker service."
        exit 1
    fi
}

# ── Build ─────────────────────────────────────────────────────────────────────
do_build() {
    check_docker

    log "Building image  ${IMAGE_NAME}:${IMAGE_TAG}  (native linux/amd64 with GPU)"
    log "GPU-accelerated build: all CUDA extensions compiled at build time"
    log "This will take a while (45-90+ min on first build)..."
    echo ""

    docker build \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"

    ok "Image built:  ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    log "Quick test:"
    echo "   docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')\""
}

# ── Export ────────────────────────────────────────────────────────────────────
do_export() {
    check_docker

    log "Exporting image to  ${TAR_FILE}..."
    log "This may take 10-20 min depending on image size."

    docker save -o "${SCRIPT_DIR}/${TAR_FILE}" "${IMAGE_NAME}:${IMAGE_TAG}"

    SIZE=$(du -h "${SCRIPT_DIR}/${TAR_FILE}" | cut -f1)
    ok "Exported:  ${TAR_FILE}  (${SIZE})"
    echo ""
    log "Transfer to your GPU server and load with:"
    echo "   docker load -i ${TAR_FILE}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
CMD="${1:-all}"

case "${CMD}" in
    build)
        do_build
        ;;
    export)
        do_export
        ;;
    all)
        do_build
        do_export
        ;;
    *)
        echo "Usage: $0 [build|export|all]"
        echo ""
        echo "  build   — Build the Docker image (GPU-enabled)"
        echo "  export  — Save image to .tar"
        echo "  all     — Build + export (default)"
        exit 1
        ;;
esac
