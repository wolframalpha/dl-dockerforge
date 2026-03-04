# =============================================================================
# Deep Learning & LLM Toolkit — Local GPU Build (Windows/WSL2 + Docker Desktop)
# Target: CUDA 12.8 | RTX 4060 (SM 8.9) | Python 3.12 (via Conda)
# Build:  docker build -t dl-toolkit-gpu .
# =============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# ── Prevent interactive prompts during build ──────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# ── System packages + OpenCV / multimedia / .so dependencies ──────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe -y && \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # ── Build essentials ──
    build-essential cmake ninja-build pkg-config autoconf automake libtool \
    git git-lfs curl wget unzip zip ca-certificates gnupg \
    lsb-release \
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
    # ── OpenCV C++ libs (Python bindings installed via pip, NOT apt) ──
    libopencv-dev \
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
    libnuma-dev libgl1 libegl1-mesa libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    libcurl4-openssl-dev libmagic1 \
    # ── PDF rendering & text extraction ──
    poppler-utils libpoppler-dev \
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

# =============================================================================
#  Conda (Miniforge) — single Python 3.12 for ALL packages
# =============================================================================
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# Conda FIRST on PATH — this is now the ONE Python
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Pin Python 3.12, enable base by default, clean up
RUN conda install -y python=3.12 && \
    conda init bash && \
    conda config --set auto_activate_base true && \
    conda clean -afy

# ── Symlink conda's python to /usr/local/bin for any tool that looks there ────
RUN ln -sf ${CONDA_DIR}/bin/python /usr/local/bin/python && \
    ln -sf ${CONDA_DIR}/bin/python /usr/local/bin/python3 && \
    ln -sf ${CONDA_DIR}/bin/python /usr/local/bin/python3.12

# ── Install uv (blazing fast Python package manager) ──────────────────────────
RUN pip install --no-cache-dir uv

# ── Tell uv to always target Conda's Python ──────────────────────────────────
ENV UV_PYTHON=${CONDA_DIR}/bin/python

# ── CUDA environment ──────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# RTX 4060 = SM 8.9 (Ada Lovelace), also include 8.0 (A100) for server compat
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9"

# =============================================================================
#  Python packages — installed with uv into Conda's Python 3.12
# =============================================================================
WORKDIR /workspace

# ── PyTorch (latest stable, CUDA 12.8) ────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ── vLLM 0.17.0+ (nightly wheels) ─────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

# ── Hugging Face ecosystem ────────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
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
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    unsloth

# ── Quantization & optimization ──────────────────────────────────────────────
# bitsandbytes & optimum don't need torch at build time
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    bitsandbytes \
    optimum

# NOTE: auto-gptq (archived) and autoawq (deprecated) removed.
# GPTQ quantization is natively supported by vLLM (Marlin kernels) and
# HuggingFace transformers — no standalone package needed.

# ── Triton & xformers ─────────────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    triton \
    xformers

# ── DeepSpeed ─────────────────────────────────────────────────────────────────
# RUN DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=1 \
#     uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
#     deepspeed

# ── Fix Conda linker conflict with CUDA ───────────────────────────────────────
# Conda's compiler_compat/ld shadows the system linker and can't find libcuda.so.1.
# Remove it so the system ld is used. CUDA stubs provide libcuda.so for link-time.
RUN rm -f ${CONDA_DIR}/compiler_compat/ld
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

# ── llama.cpp — fully built with CUDA at build time ──────────────────────────
# Python bindings with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" MAX_JOBS=4 \
    uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir llama-cpp-python

# Clone and build llama.cpp CLI with CUDA
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;89" && \
    cmake --build build --config Release -j4 && \
    cp build/bin/* /usr/local/bin/ 2>/dev/null || true
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# ── LangChain & LlamaIndex ────────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    langchain \
    langchain-community \
    langchain-core \
    langchain-openai \
    llama-index

# ── Embeddings & Vector stores ────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    sentence-transformers \
    chromadb

# faiss-gpu-cu12 now has Python 3.12 + CUDA 12.x wheels
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir faiss-gpu-cu12

# ── OpenCV (Python) ───────────────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    opencv-python-headless \
    opencv-contrib-python-headless

# ── Experiment tracking & notebooks ───────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    wandb \
    tensorboard \
    mlflow \
    jupyterlab \
    notebook \
    ipywidgets

# ── Data science & utilities ──────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
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
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    ultralytics

# ── SGLang (inference engine) ─────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    "sglang[all]"

# ── MPI for Python + GPU monitoring ──────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    mpi4py \
    pynvml \
    packaging

# ── ONNX & TensorRT ──────────────────────────────────────────────────────────
RUN uv pip install --python ${CONDA_DIR}/bin/python --no-cache-dir \
    onnx onnxruntime-gpu

# ── Final verification (GPU available — all checks should pass) ───────────────
RUN python -c 'import sys; print(f"Python {sys.version}")' && \
    python -c 'import torch; print(f"PyTorch {torch.__version__}"); print(f"CUDA compiled: {torch.version.cuda}"); print(f"GPU available: {torch.cuda.is_available()}")' && \
    python -c 'import vllm; print(f"vLLM {vllm.__version__}")' && \
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
