# DL Toolkit — Local GPU Build (Windows/WSL2)

> **Target**: CUDA 12.8 · RTX 4060 (SM 8.9) + A100 (SM 8.0) · Python 3.12  
> **Build with**: Podman on Windows GPU laptop  
> **Key difference from Mac build**: ALL CUDA extensions compiled at build time — zero post-boot setup  
> **Services**: JupyterLab + SSH (VS Code Remote) running simultaneously

---

## What's Different from `docker_local_mac`?

| Feature | `docker_local_mac` | `docker_local_gpu` |
|---------|--------------------|--------------------|
| Build host | Mac (Podman, no GPU) | Windows (Podman, with GPU) |
| Flash Attention | Download-only + post-boot script | ✅ Compiled at build time |
| bitsandbytes | `BUILD_CUDA_EXT=0` (CPU) | ✅ Full CUDA build |
| AutoGPTQ / AutoAWQ | `BUILD_CUDA_EXT=0` (CPU) | ✅ Full CUDA build |
| llama.cpp | CPU-only + post-boot script | ✅ Built with CUDA |
| DeepSpeed ops | Python-only | ✅ `DS_BUILD_OPS=1` |
| SGLang | `sglang[all]` | ✅ `sglang[all]>=0.5.9` + FlashInfer |
| Post-boot scripts | 2 scripts required | ❌ None needed |

---

## What's Inside

| Category | Libraries |
|----------|-----------|
| **Core** | PyTorch (CUDA 12.8), Triton, xformers |
| **LLM Inference** | vLLM (nightly), SGLang (0.5.9+), llama.cpp (CUDA) |
| **Fine-tuning** | Unsloth, PEFT, TRL, DeepSpeed (compiled) |
| **HF Ecosystem** | Transformers, Accelerate, Datasets, Tokenizers, Safetensors |
| **Quantization** | bitsandbytes (CUDA), AutoGPTQ (CUDA), AutoAWQ (CUDA), Optimum |
| **Attention** | Flash Attention 2 (compiled) |
| **RAG / Agents** | LangChain, LlamaIndex, Sentence-Transformers, FAISS, ChromaDB |
| **Vision** | OpenCV (headless + contrib), Ultralytics (YOLO), ffmpeg |
| **Serving** | FastAPI, Uvicorn, Gradio, Streamlit |
| **Tracking** | W&B, TensorBoard, MLflow, JupyterLab |
| **Data Science** | NumPy, Pandas, Polars, Scikit-learn, SciPy, Matplotlib, Seaborn, Plotly |
| **Multi-GPU** | NCCL, OpenMPI, InfiniBand/RDMA libs |
| **Runtime** | ONNX Runtime GPU, conda (Miniforge) |

---

## Prerequisites (Windows)

### 1. Install Podman

```powershell
# Install Podman via winget
winget install RedHat.Podman

# Initialize and start the Podman machine (allocate enough resources)
podman machine init --cpus 4 --memory 8192 --disk-size 100
podman machine start
```

> ⚠️ Set `--disk-size` to at least **100 GB** — the image is large.

### 2. Install NVIDIA drivers + CUDA 12.8

```powershell
# Check your current driver
nvidia-smi
# Update to latest Game Ready or Studio Driver from nvidia.com
```

### 3. Verify GPU access in Podman

```bash
# In WSL2 terminal or PowerShell:
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### 1. Build the image (Podman on Windows)

```bash
chmod +x build.sh
./build.sh build
```

> ⚠️ First build takes **45–90+ minutes**. CUDA compilation is CPU-intensive even with GPU present.  
> The image will be **30–50 GB**.

### 2. Export to .tar for transfer

```bash
./build.sh export
# or do both in one command:
./build.sh all
```

### 3. Transfer to GPU server

```bash
scp dl-toolkit-gpu.tar user@gpu-server:/path/to/destination/
```

### 4. Load & run on the GPU server (Docker on server)

```bash
# Load the image (server uses Docker)
docker load -i dl-toolkit-gpu.tar

# Run with GPU access
docker run -d \
    --gpus all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name dl-toolkit-gpu \
    -v /data:/workspace/data \
    -e ROOT_PASSWORD=your_secure_password \
    -e JUPYTER_PASSWORD=your_password \
    dl-toolkit-gpu:latest
```

> **🎉 No post-boot setup needed!** Everything is pre-compiled and ready to use.

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

### Drop into a bash shell (on the GPU server)

```bash
docker exec -it dl-toolkit-gpu /bin/bash
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

## Example Workflows

```bash
# Serve a model with vLLM
vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2

# Serve with SGLang
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --tp 2

# llama.cpp inference (CUDA-accelerated)
llama-cli -m model.gguf -p "Hello" -ngl 99

# Fine-tune with Unsloth
python train.py  # your training script
```

---

## Customisation

```bash
IMAGE_NAME=dltoolkit-gpu IMAGE_TAG=deviprasad ./build.sh all
```
