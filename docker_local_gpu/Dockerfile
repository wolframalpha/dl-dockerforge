# =============================================================================
# Deep Learning & LLM Toolkit — Local GPU Build (Windows/WSL2 + Docker Desktop)
# Target: CUDA 12.8 | RTX 4060 (SM 8.9) | Python 3.12
# Build:  docker build -t dl-toolkit-gpu .
# =============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

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
    # ── Multi-GPU: NCCL (GPU-GPU communication) ──
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
ENV PATH="${PATH}:${CONDA_DIR}/bin"
RUN conda init bash && conda config --set auto_activate_base false && \
    conda clean -afy

# ── Force uv to use our Python 3.12, not conda's Python ──────────────────────
ENV UV_PYTHON=/usr/bin/python3.12

# ── CUDA environment ──────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# RTX 4060 = SM 8.9 (Ada Lovelace), also include 8.0 (A100) for server compat
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9"

# =============================================================================
#  Python packages — installed with uv for speed
#  ALL CUDA extensions are compiled at build time (GPU available in builder)
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
# Full CUDA extension build — GPU available during build
RUN MAX_JOBS=4 \
    uv pip install --system --no-cache-dir \
    bitsandbytes \
    auto-gptq \
    optimum \
    autoawq

# ── Flash Attention — compiled with CUDA at build time ────────────────────────
RUN MAX_JOBS=4 uv pip install --system --no-cache-dir \
    flash-attn --no-build-isolation

# ── Triton & xformers ─────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    triton \
    xformers

# ── DeepSpeed ─────────────────────────────────────────────────────────────────
RUN DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=1 \
    uv pip install --system --no-cache-dir \
    deepspeed

# ── llama.cpp — fully built with CUDA at build time ──────────────────────────
# Python bindings with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" MAX_JOBS=4 \
    uv pip install --system --no-cache-dir llama-cpp-python

# Clone and build llama.cpp CLI with CUDA
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;89" && \
    cmake --build build --config Release -j4 && \
    cp build/bin/* /usr/local/bin/ 2>/dev/null || true
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

# ── ONNX & TensorRT ──────────────────────────────────────────────────────────
RUN uv pip install --system --no-cache-dir \
    onnx onnxruntime-gpu

# ── Final verification (GPU available — all checks should pass) ───────────────
RUN python -c 'import sys; print(f"Python {sys.version}")' && \
    python -c 'import torch; print(f"PyTorch {torch.__version__}"); print(f"CUDA compiled: {torch.version.cuda}"); print(f"GPU available: {torch.cuda.is_available()}")' && \
    python -c 'import vllm; print(f"vLLM {vllm.__version__}")' && \
    python -c 'import flash_attn; print(f"Flash Attention {flash_attn.__version__}")' && \
    python -c 'import sglang; print(f"SGLang {sglang.__version__}")' && \
    python -c 'import bitsandbytes; print(f"bitsandbytes {bitsandbytes.__version__}")' && \
    python -c 'import deepspeed; print(f"DeepSpeed {deepspeed.__version__}")' \
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
