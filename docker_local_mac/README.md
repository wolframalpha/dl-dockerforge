# DL Toolkit — Offline GPU Server Docker Image

> **Target**: CUDA 12.8 · 2× A100 · Python 3.12  
> **Build with**: Podman (Mac) → transfer .tar → load with Docker (GPU server)  
> **Services**: JupyterLab + SSH (VS Code Remote) running simultaneously

---

## What's Inside

| Category | Libraries |
|----------|-----------|
| **Core** | PyTorch (CUDA 12.8), Triton, xformers |
| **LLM Inference** | vLLM (nightly), llama.cpp (CLI + Python, CUDA-accelerated) |
| **Fine-tuning** | Unsloth, PEFT, TRL, DeepSpeed |
| **HF Ecosystem** | Transformers, Accelerate, Datasets, Tokenizers, Safetensors |
| **Quantization** | bitsandbytes, AutoGPTQ, AutoAWQ, Optimum |
| **Attention** | Flash Attention 2 |
| **RAG / Agents** | LangChain, LlamaIndex, Sentence-Transformers, FAISS-GPU, ChromaDB |
| **Vision** | OpenCV (headless + contrib), system codec libs, ffmpeg |
| **Serving** | FastAPI, Uvicorn, Gradio, Streamlit |
| **Tracking** | W&B, TensorBoard, MLflow, JupyterLab |
| **Data Science** | NumPy, Pandas, Polars, Scikit-learn, SciPy, Matplotlib, Seaborn, Plotly |
| **Multi-GPU** | NCCL, OpenMPI, InfiniBand/RDMA libs |
| **Runtime** | ONNX Runtime GPU, conda (Miniforge) |

---

## Prerequisites (Mac)

```bash
# Install Podman if you don't have it
brew install podman

# Initialize and start the Podman machine
podman machine init --cpus 4 --memory 8192 --disk-size 100
podman machine start
```

> ⚠️ Set `--disk-size` to at least **100 GB** — the image is large.

---

## Quick Start

### 1. Build the image (Podman on Mac)

```bash
chmod +x build.sh
./build.sh build
```

> ⚠️ First build takes **30–60+ minutes** and the image will be **30–50 GB**.

### 2. Export to .tar for transfer

```bash
./build.sh export
# or do both in one command:
./build.sh all
```

### 3. Transfer to GPU server

```bash
# The .tar is in docker-archive format — compatible with Docker
scp dl-toolkit.tar user@gpu-server:/path/to/destination/
```

### 4. Load & run on the GPU server (Docker)

```bash
# Load the image
docker load -i dl-toolkit.tar

# Run with ALL host ports mapped + GPU access
docker run -d \
    --gpus all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name dl-toolkit \
    -v /data:/workspace/data \
    -e ROOT_PASSWORD=your_secure_password \
    -e JUPYTER_PASSWORD=your_password \
    dl-toolkit:latest
```

> `--net=host` maps **all** container ports directly to host ports — no filtering.  
> `--ipc=host` is required for PyTorch multi-GPU shared memory.

### 5. ⚡ First boot — compile CUDA extensions (one-time, no internet needed)

Some libraries can't compile CUDA kernels on Mac, so they ship CPU-only in the image. Run these **once** after first boot on the GPU server:

```bash
docker exec -it dl-toolkit bash

# 1. Compile Flash Attention (~5 min)
/opt/install-flash-attn.sh

# 2. Compile llama.cpp + llama-cpp-python with CUDA (~3 min)
/opt/install-llama-cuda.sh
```

> These scripts use source code already bundled in the image — **zero internet required**.  
> You only need to run them once. After that, everything is GPU-accelerated.

---

## Accessing the Container

### JupyterLab (browser)

```
http://<gpu-server-ip>:8888/lab
```

Enter the password you set via `JUPYTER_PASSWORD` (default: `dltoolkit`).

### VS Code Remote-SSH

1. Install the **Remote - SSH** extension in VS Code
2. Add to your SSH config (`~/.ssh/config`):

```
Host gpu-container
    HostName <gpu-server-ip>
    User root
    Port 2222
```

3. Connect via **Remote-SSH: Connect to Host → gpu-container**
4. Password: whatever you set in `ROOT_PASSWORD` (default: `dltoolkit`)

> **Tip**: To use SSH keys instead of password:
> ```bash
> # From your local machine
> ssh-copy-id -p 2222 root@<gpu-server-ip>
> ```

### Drop into a bash shell

```bash
docker exec -it dl-toolkit /bin/bash
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_PASSWORD` | `dltoolkit` | SSH root password |
| `SSH_PORT` | `2222` | SSH port (avoids conflict with host's port 22) |
| `JUPYTER_PORT` | `8888` | JupyterLab port |
| `JUPYTER_PASSWORD` | `dltoolkit` | JupyterLab login password |

---

## Key Ports (all mapped via `--net=host`)

| Port | Service |
|------|---------|
| 2222 | SSH (VS Code Remote) — avoids conflict with host SSHD on 22 |
| 8888 | JupyterLab |
| 8000 | vLLM / FastAPI |
| 7860 | Gradio |
| 5000 | MLflow |

---

## Example workflows inside the container

```bash
# Serve a model with vLLM (uses both A100s)
vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2

# llama.cpp inference
llama-cli -m model.gguf -p "Hello" -ngl 99

# Fine-tune with Unsloth
python train.py  # your training script
```

---

## Customisation

```bash
IMAGE_NAME=dltoolkit IMAGE_TAG=deviprasad ./build.sh all
```
