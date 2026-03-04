# =============================================================================
# Deep Learning & LLM Toolkit — Offline GPU Server Image
# Target: CUDA 12.8 | 2× A100 | Python 3.12
# Build:  docker buildx build --platform linux/amd64 -t dl-toolkit .
# =============================================================================

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-devel-ubuntu22.04

# ── Prevent interactive prompts during build ──────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# ── System packages + OpenCV / multimedia / .so dependencies ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ── Build essentials ──
    build-essential cmake ninja-build pkg-config autoconf automake libtool \
    git git-lfs curl wget unzip zip ca-certificates gnupg \
    software-properties-common lsb-release \
    # ── Python 3.12 via deadsnakes PPA ──
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    python3.12 python3.12-dev python3.12-venv \
    # ── Multi-GPU: NCCL (critical for 2× A100 communication) ──
    libnccl2 libnccl-dev \
    # ── MPI (required by DeepSpeed for distributed training) ──
    libopenmpi-dev openmpi-bin openmpi-common \
    # ── InfiniBand / RDMA (fast GPU-GPU interconnect) ──
    libibverbs-dev librdmacm-dev ibverbs-providers rdma-core \
    # ── SSH (multi-node training, remote debugging) ──
    openssh-client openssh-server \
    # ── DeepSpeed dependencies ──
    libaio-dev pdsh \
    # ── OpenCV + image/video codec dependencies ──
    libopencv-dev python3-opencv \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libx265-dev \
    ffmpeg \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libgtk-3-dev libcanberra-gtk3-module \
    # ── Linear algebra & math ──
    libopenblas-dev liblapack-dev libatlas-base-dev gfortran \
    # ── HDF5 & compression ──
    libhdf5-dev liblz4-dev libzstd-dev libbz2-dev \
    # ── Audio ──
    libsndfile1-dev libsox-dev sox \
    # ── Database clients (MLflow, experiment logging) ──
    libpq-dev libsqlite3-dev \
    # ── Misc .so files commonly needed by Python packages ──
    libffi-dev libssl-dev libxml2-dev libxslt1-dev \
    libnuma-dev libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    libcurl4-openssl-dev libmagic1 \
    # ── Locale (UTF-8 support) ──
    locales \
    # ── Monitoring & debugging ──
    numactl htop tmux screen vim nano \
    pciutils net-tools iputils-ping dnsutils \
    strace ltrace gdb valgrind \
    jq tree less file sudo \
    && rm -rf /var/lib/apt/lists/*

# ── GitHub CLI (gh) ───────────────────────────────────────────────────────────
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# ── Initialize Git LFS ───────────────────────────────────────────────────────
RUN git lfs install

# ── Generate UTF-8 locale ─────────────────────────────────────────────────────
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ── Configure SSH (needed for DeepSpeed multi-node) ───────────────────────────
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# ── Set Python 3.12 as default ────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# ── Install uv (blazing fast Python package manager) ──────────────────────────
RUN pip install --no-cache-dir uv

# ── Install Miniforge (conda) ────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh
# Put conda at END of PATH so system python3.12 takes priority
# Conda is still available via `conda activate` for separate envs
ENV PATH="${PATH}:${CONDA_DIR}/bin"
RUN conda init bash && conda config --set auto_activate_base false && \
    conda clean -afy

# ── Force uv to use our Python 3.12, not conda's Python ──────────────────────
ENV UV_PYTHON=/usr/bin/python3.12

# ── CUDA environment ──────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.0"

# =============================================================================
#  Python packages — installed with uv for speed
# =============================================================================
WORKDIR /workspace

# ── PyTorch (latest stable, CUDA 12.8) ────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ── vLLM 0.17.0+ (nightly wheels) ─────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

# ── Hugging Face ecosystem ────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    peft \
    trl \
    evaluate \
    huggingface_hub[cli]

# ── Unsloth ───────────────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    unsloth

# ── Quantization & optimization ──────────────────────────────────────────────
# BUILD_CUDA_EXT=0: skip CUDA compilation on Mac; extensions JIT-compile on GPU
# MAX_JOBS=2: prevent OOM during builds in Podman
RUN BUILD_CUDA_EXT=0 MAX_JOBS=2 \
    uv pip install --system --no-cache-dir \
    bitsandbytes \
    auto-gptq \
    optimum \
    autoawq

# ── Flash Attention ───────────────────────────────────────────────────────────
# Can't compile on Mac (OOM + no GPU). Download source now, compile on GPU server.
RUN uv pip download --no-cache-dir flash-attn -d /opt/flash-attn-wheels 2>/dev/null || \
    pip download flash-attn --no-deps -d /opt/flash-attn-wheels 2>/dev/null || \
    (git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention && \
     echo "Source cloned for local build")
# On the GPU server, run:  /opt/install-flash-attn.sh
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Installing flash-attn (compiling CUDA kernels)..."\n\
if ls /opt/flash-attn-wheels/flash_attn*.whl 1>/dev/null 2>&1; then\n\
    pip install /opt/flash-attn-wheels/flash_attn*.whl\n\
elif ls /opt/flash-attn-wheels/flash*.tar.gz 1>/dev/null 2>&1; then\n\
    MAX_JOBS=4 pip install /opt/flash-attn-wheels/flash*.tar.gz --no-build-isolation\n\
elif [ -d /opt/flash-attention ]; then\n\
    cd /opt/flash-attention && MAX_JOBS=4 pip install . --no-build-isolation\n\
fi\n\
echo "flash-attn installed successfully!"' > /opt/install-flash-attn.sh && \
    chmod +x /opt/install-flash-attn.sh

# ── Triton & xformers ─────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    triton \
    xformers

# ── DeepSpeed ─────────────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    deepspeed

# ── llama.cpp ─────────────────────────────────────────────────────────────────
# CPU-only Python bindings (CUDA compilation OOMs on Mac)
RUN uv pip install --system --no-cache-dir llama-cpp-python

# Clone llama.cpp source (for GPU recompilation on server)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp

# Script to recompile everything with CUDA on the GPU server (no internet needed)
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Recompiling llama-cpp-python with CUDA..."\n\
CMAKE_ARGS="-DGGML_CUDA=on" MAX_JOBS=4 pip install llama-cpp-python --force-reinstall --no-cache-dir --no-deps\n\
echo "Building llama.cpp CLI with CUDA..."\n\
cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 && \\\n\
cmake --build build --config Release -j4 && \\\n\
cp build/bin/* /usr/local/bin/ 2>/dev/null || true\n\
echo "llama.cpp CUDA build complete!"' > /opt/install-llama-cuda.sh && \
    chmod +x /opt/install-llama-cuda.sh
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# ── LangChain & LlamaIndex ────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    langchain \
    langchain-community \
    langchain-core \
    langchain-openai \
    llama-index

# ── Embeddings & Vector stores ────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    sentence-transformers \
    chromadb

# faiss-gpu has no Python 3.12 pip wheels; using faiss-cpu (still fast for indexing)
# Conda faiss-gpu breaks pip-installed PyTorch — avoid mixing package managers
RUN uv pip install --system --no-cache-dir faiss-cpu

# ── OpenCV (Python) ───────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    opencv-python-headless \
    opencv-contrib-python-headless

# ── Experiment tracking & notebooks ───────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    wandb \
    tensorboard \
    mlflow \
    jupyterlab \
    notebook \
    ipywidgets

# ── Data science & utilities ──────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    numpy scipy pandas polars \
    scikit-learn \
    matplotlib seaborn plotly \
    einops \
    tiktoken \
    sentencepiece \
    protobuf \
    pyyaml \
    tqdm \
    rich \
    typer \
    httpx \
    fastapi uvicorn \
    gradio \
    streamlit

# ── Computer Vision (YOLO) ────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    ultralytics

# ── SGLang (inference engine) ─────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    "sglang[all]"

# ── MPI for Python + GPU monitoring ──────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    mpi4py \
    pynvml \
    packaging

# ── NVIDIA Optimum backend ────────────────────────────────────────────────────
# SKIPPED: optimum-nvidia's dependency solver uninstalls torch during build,
# and it requires TensorRT wheels from pypi.nvidia.com (needs internet).

# ── ONNX & TensorRT ──────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    onnx onnxruntime-gpu

# ── Final verification (non-fatal on Mac — no GPU available) ──────────────────
RUN python -c 'import sys; print(f"Python {sys.version}")' && \
    python -c 'import torch; print(f"PyTorch {torch.__version__}"); print(f"CUDA compiled: {torch.version.cuda}")' && \
    python -c 'import vllm; print(f"vLLM {vllm.__version__}")' \
    || true

# ── SSH host keys (generate once at build time) ───────────────────────────────
RUN ssh-keygen -A

# ── Entrypoint: runs SSHD + JupyterLab simultaneously ────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Environment defaults (override at runtime with -e) ───────────────────────
ENV ROOT_PASSWORD=dltoolkit
ENV SSH_PORT=2222
ENV JUPYTER_PORT=8888
ENV JUPYTER_PASSWORD=dltoolkit

# ── All ports are mapped via --net=host, but document key ones ────────────────
# 2222  → SSH (VS Code Remote-SSH) — avoids conflict with host's SSHD on 22
# 8888  → JupyterLab
# 8000  → vLLM / FastAPI
# 7860  → Gradio
# 5000  → MLflow
EXPOSE 2222 8000 8888 7860 5000

ENTRYPOINT ["/entrypoint.sh"]
