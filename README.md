<p align="center">
  <h1 align="center">🐳 dl-dockerforge</h1>
  <p align="center">
    <b>Production-ready Docker images for deep learning & LLM workloads on air-gapped GPU servers</b>
  </p>
  <p align="center">
    Build on your laptop. Ship a <code>.tar</code>. Run on an offline GPU server. Zero internet required at runtime.
  </p>
</p>

<p align="center">
  <a href="#-whats-inside">What's Inside</a> •
  <a href="#-the-problem">The Problem</a> •
  <a href="#-build-variants">Build Variants</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-access-your-environment">Access</a> •
  <a href="#-configuration">Configuration</a>
</p>

---

## 🧠 The Problem

You've got a powerful GPU server (A100s, RTX 4090s, etc.) sitting in a data center — but it has **no internet access**. You need PyTorch, vLLM, Flash Attention, DeepSpeed, and 60+ other libraries, all compiled with CUDA extensions. Building from scratch on an air-gapped machine is painful.

**dl-dockerforge** solves this:

1. **Build** the Docker image on your local machine (Mac or Windows GPU)
2. **Export** to a `.tar` file
3. **Transfer** via `scp` / USB / sneakernet
4. **Load** on the GPU server — everything just works, GPU-accelerated, no internet needed

---

## 📦 What's Inside

A **batteries-included** deep learning environment with 60+ pre-installed libraries:

| Category | Libraries |
|----------|-----------|
| **Core** | PyTorch (CUDA 12.8), Triton, xformers |
| **LLM Inference** | vLLM (nightly), SGLang, llama.cpp (CUDA) |
| **Fine-tuning** | Unsloth, PEFT, TRL, DeepSpeed |
| **HF Ecosystem** | Transformers, Accelerate, Datasets, Tokenizers, Safetensors |
| **Quantization** | bitsandbytes, AutoGPTQ, AutoAWQ, Optimum |
| **Attention** | Flash Attention 2 |
| **RAG / Agents** | LangChain, LlamaIndex, Sentence-Transformers, FAISS, ChromaDB |
| **Computer Vision** | OpenCV, Ultralytics (YOLO), ffmpeg |
| **Serving** | FastAPI, Uvicorn, Gradio, Streamlit |
| **Experiment Tracking** | W&B, TensorBoard, MLflow |
| **Data Science** | NumPy, Pandas, Polars, Scikit-learn, SciPy, Matplotlib, Seaborn, Plotly |
| **Multi-GPU** | NCCL, OpenMPI, InfiniBand/RDMA libs |
| **Runtime** | ONNX Runtime GPU, Miniforge (conda), JupyterLab |

**Services** running on boot:
- 🔬 **JupyterLab** — browser-based notebooks
- 🔌 **SSH server** — connect via VS Code Remote-SSH

---

## 🔀 Build Variants

Choose the variant that matches your build machine:

### [`docker_local_mac/`](docker_local_mac/) — Build on Mac, deploy to GPU server

> **Best for**: You develop on a MacBook but run training on a remote GPU server

- Cross-compiles `linux/amd64` image via Podman on macOS
- CUDA extensions that can't compile without a GPU (Flash Attention, llama.cpp) are bundled as source — compiled on the server with included post-boot scripts (`/opt/install-flash-attn.sh`, `/opt/install-llama-cuda.sh`)
- **One-time post-boot setup (~8 min)**, then everything is GPU-accelerated

### [`docker_local_gpu/`](docker_local_gpu/) — Build on a GPU machine (Windows/WSL2)

> **Best for**: You have a local Windows machine with an NVIDIA GPU

- Builds natively on GPU — **all CUDA extensions pre-compiled** at build time
- Flash Attention, bitsandbytes, DeepSpeed, llama.cpp — all compiled with CUDA
- **Zero post-boot setup** — run the container and start working immediately

### Comparison

| Feature | `docker_local_mac` | `docker_local_gpu` |
|---------|--------------------|--------------------|
| **Build host** | Mac (no GPU) | Windows/WSL2 (with GPU) |
| **Flash Attention** | Post-boot compile | ✅ Pre-compiled |
| **bitsandbytes** | CPU build | ✅ CUDA build |
| **AutoGPTQ / AutoAWQ** | CPU build | ✅ CUDA build |
| **llama.cpp** | Post-boot compile | ✅ CUDA build |
| **DeepSpeed ops** | Python-only | ✅ `DS_BUILD_OPS=1` |
| **Post-boot scripts** | 2 scripts required | ❌ None needed |
| **Build time** | ~30–60 min | ~45–90 min |

---

## 🚀 Quick Start

### Prerequisites

<details>
<summary><b>Mac (Podman)</b></summary>

```bash
brew install podman
podman machine init --cpus 4 --memory 8192 --disk-size 100
podman machine start
```

> ⚠️ Set `--disk-size` to at least **100 GB** — the final image is 30–50 GB.
</details>

<details>
<summary><b>Windows/WSL2 (Podman)</b></summary>

```powershell
winget install RedHat.Podman
podman machine init --cpus 4 --memory 8192 --disk-size 100
podman machine start
```

Verify GPU access:
```bash
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
</details>

### Build & Export

```bash
# Pick your variant
cd docker_local_mac   # or docker_local_gpu

# Build the image and export to .tar
chmod +x build.sh
./build.sh all
```

### Transfer & Run on GPU Server

```bash
# Transfer the .tar to your GPU server
scp dl-toolkit.tar user@gpu-server:/path/to/

# On the GPU server — load and run
docker load -i dl-toolkit.tar

docker run -d \
    --gpus all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name dl-toolkit \
    -v /data:/workspace/data \
    -e ROOT_PASSWORD=changeme \
    -e JUPYTER_PASSWORD=changeme \
    dl-toolkit:latest
```

### Post-boot setup (Mac build only)

```bash
docker exec -it dl-toolkit bash

# Compile Flash Attention (~5 min)
/opt/install-flash-attn.sh

# Compile llama.cpp with CUDA (~3 min)
/opt/install-llama-cuda.sh
```

> GPU build users: skip this — everything is already compiled.

---

## 💻 Access Your Environment

### JupyterLab

```
http://<gpu-server-ip>:8888/lab
```

### VS Code Remote-SSH

Add to `~/.ssh/config`:

```
Host gpu-container
    HostName <gpu-server-ip>
    User root
    Port 2222
```

Connect via **Remote-SSH → gpu-container** (password: your `ROOT_PASSWORD`).

### Bash Shell

```bash
docker exec -it dl-toolkit /bin/bash
```

---

## ⚙️ Configuration

All settings are configurable via environment variables at `docker run` time:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_PASSWORD` | `dltoolkit` | SSH root password |
| `SSH_PORT` | `2222` | SSH port (avoids conflict with host SSHD on 22) |
| `JUPYTER_PORT` | `8888` | JupyterLab port |
| `JUPYTER_PASSWORD` | `dltoolkit` | JupyterLab login password |

### Exposed Ports

| Port | Service |
|------|---------|
| `2222` | SSH (VS Code Remote) |
| `8888` | JupyterLab |
| `8000` | vLLM / FastAPI |
| `7860` | Gradio |
| `5000` | MLflow |

---

## 🧪 Example Workflows

```bash
# Serve a model with vLLM (multi-GPU)
vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2

# Serve with SGLang
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --tp 2

# GGUF inference with llama.cpp (CUDA-accelerated)
llama-cli -m model.gguf -p "Hello" -ngl 99

# Fine-tune with Unsloth
python train.py
```

---

## 📁 Repository Structure

```
dl-dockerforge/
├── docker_local_mac/       # Mac → GPU server workflow
│   ├── Dockerfile          # Cross-compiled linux/amd64 image
│   ├── build.sh            # Build & export script (Podman)
│   ├── entrypoint.sh       # JupyterLab + SSHD launcher
│   └── README.md           # Mac-specific instructions
│
├── docker_local_gpu/       # Windows GPU workflow
│   ├── Dockerfile          # Native GPU build, all CUDA compiled
│   ├── build.sh            # Build & export script (Podman)
│   ├── entrypoint.sh       # JupyterLab + SSHD launcher
│   └── README.md           # GPU-specific instructions
│
└── README.md               # This file
```

---

## 🎯 Design Decisions

- **Podman over Docker for builds** — rootless, daemonless, and runs natively on Mac without Docker Desktop licensing
- **`uv` for Python packages** — 10-50x faster installs than pip
- **Miniforge (conda) included but not default** — available for niche packages without polluting the main Python environment
- **`--net=host` networking** — simplest approach for GPU servers; all ports directly accessible
- **CUDA 12.8 + Python 3.12** — latest stable stack as of 2025

---

## 🤝 Contributing

Contributions welcome! Some ideas:

- [ ] Add a `docker_cloud/` variant for cloud GPU builds (Lambda, RunPod, etc.)
- [ ] CI/CD pipeline for automated image builds
- [ ] ARM64 / Grace Hopper support
- [ ] Image size optimization with multi-stage builds

---

## 📄 License

MIT License — use it however you want.

---

<p align="center">
  <b>Built for people who just want to train models,<br/>not debug CUDA compilation errors. 🚂</b>
</p>
