#!/usr/bin/env bash
# Last updated: 2026-05-17
#
# Reproduce the Pixal3D environment built against:
#   Python 3.13 + torch 2.12.0+cu132 + Ada (sm_89, RTX 4090) + CUDA toolkit 13.2
#
# This is NOT the upstream-recommended stack. Upstream (TRELLIS.2 setup.sh)
# uses Python 3.10 + torch 2.6 + cu124 and ships prebuilt cp310 wheels.
# If you can use that combination, the prebuilt wheels in
# requirements-hfdemo.txt will install in seconds. This script exists for
# environments where downgrading is not an option — every CUDA extension
# is rebuilt from source.
#
# If you have prebuilt wheels for THIS exact toolchain (Python 3.13 +
# torch 2.12 + cu132 + sm_89) cached in ./internal/wheels/, they will be
# preferred over rebuilding. That directory is gitignored.
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - A CUDA toolkit matching the torch build (CUDA 13.x for torch 2.12)
#   - A GPU with compute capability >= 7.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ----------------------------------------------------------------------
# 1. Virtualenv
# ----------------------------------------------------------------------
if [ ! -d .venv ]; then
    uv venv --python 3.13
fi
PY="$REPO_ROOT/.venv/bin/python"
export VIRTUAL_ENV="$REPO_ROOT/.venv"

# ----------------------------------------------------------------------
# 2. Torch + base Python deps
#    Torch index is pytorch's cu132 wheels (which match this CUDA toolkit).
#    Falls back to default PyPI if the cu132 index isn't reachable.
# ----------------------------------------------------------------------
uv pip install --python "$PY" \
    --index-url https://download.pytorch.org/whl/cu132 \
    torch==2.12.0 torchvision==0.27.0 triton==3.7.0 || \
uv pip install --python "$PY" torch torchvision triton

uv pip install --python "$PY" \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh \
    transformers==4.57.3 timm==1.0.22 kornia==0.8.2 zstandard \
    diffusers==0.37.1 accelerate==1.13.0 gradio plyfile==1.1.3 \
    ninja huggingface_hub safetensors einops

# ----------------------------------------------------------------------
# 3. MoGe (depth / camera estimation) + utils3d
#    Order matters: install MoGe first (which git-pins upstream utils3d),
#    then overlay LDYang694's utils3d wheel — it adds the `pt` alias for
#    MoGe-v2 compatibility plus a `depth_map_to_point_map` helper that
#    Pixal3D's pipeline calls. Pure-Python wheel.
# ----------------------------------------------------------------------
uv pip install --python "$PY" "git+https://github.com/microsoft/MoGe.git"
uv pip install --python "$PY" --reinstall-package utils3d \
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# ----------------------------------------------------------------------
# 4. CUDA extensions — prefer cached wheels, otherwise build from source.
# ----------------------------------------------------------------------
# Override TORCH_CUDA_ARCH_LIST below for non-Ada GPUs (e.g. "8.0" for A100,
# "9.0" for H100, "12.0" for Blackwell). Leaving it unset will build for
# every arch in torch's default list, which is slow but portable.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

WHEEL_DIR="$REPO_ROOT/internal/wheels"

install_or_build () {
    local pkg="$1"
    local glob="$2"
    local src_spec="$3"

    local cached
    cached="$(ls "$WHEEL_DIR"/$glob 2>/dev/null | head -1 || true)"
    if [ -n "$cached" ]; then
        echo "[install] $pkg from cached wheel: $cached"
        uv pip install --python "$PY" "$cached"
    else
        echo "[build]   $pkg from source: $src_spec"
        uv pip install --python "$PY" --no-build-isolation "$src_spec"
    fi
}

install_or_build natten            "natten-0.21.0-*.whl"     "natten==0.21.0"
install_or_build cumesh            "cumesh-*.whl"            "git+https://github.com/JeffreyXiang/CuMesh.git"
install_or_build flex_gemm         "flex_gemm-*.whl"         "git+https://github.com/JeffreyXiang/FlexGEMM.git"
install_or_build nvdiffrast        "nvdiffrast-*.whl"        "git+https://github.com/JeffreyXiang/nvdiffrast.git"
install_or_build o_voxel           "o_voxel-*.whl"           "git+https://github.com/microsoft/TRELLIS.2.git#subdirectory=o-voxel"
install_or_build nvdiffrec_render  "nvdiffrec_render-*.whl"  "git+https://github.com/JeffreyXiang/nvdiffrec.git@renderutils"

# ----------------------------------------------------------------------
# 5. Smoke test
# ----------------------------------------------------------------------
"$PY" - <<'PY'
import torch, natten, cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render, utils3d, moge
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("all native extensions imported OK")
PY

cat <<'EOF'

Environment ready.

The TRELLIS.2 / Pixal3D flash_attn 3 wheel is sm_90-only (H100). On Ada
(RTX 4090) and earlier, set ATTN_BACKEND=sdpa to use PyTorch SDPA instead.

Run inference (replace <path-to-models> with the directory containing
pipeline.json + ckpts/, or the HF repo id "TencentARC/Pixal3D"):

  ATTN_BACKEND=sdpa .venv/bin/python inference.py \
    --image assets/images/0_img.png \
    --output ./output.glb \
    --model_path <path-to-models> \
    --low_vram --resolution 1024

Note: the default pipeline.json references the gated repo briaai/RMBG-2.0
for background removal. If you don't have access, edit your local copy
of pipeline.json and replace "briaai/RMBG-2.0" with "ZhengPeng7/BiRefNet"
(open, same architecture).
EOF
