#!/usr/bin/env bash
# Last updated: 2026-05-17
#
# Reproduce the Pixal3D environment on Python 3.13 + torch 2.12+cu132.
#
# Upstream TRELLIS.2 ships prebuilt cp310 wheels (torch 2.6 + cu124); if
# that combination is available, prefer `pip install -r requirements-hfdemo.txt`
# instead — it installs in seconds rather than minutes.
#
# Prebuilt wheels for the Python-3.13 / torch-2.12 toolchain cached in
# ./internal/wheels/ are used in preference to rebuilding. That directory
# is gitignored.
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
uv venv --python 3.13
PY="$REPO_ROOT/.venv/bin/python"
export VIRTUAL_ENV="$REPO_ROOT/.venv"

# ----------------------------------------------------------------------
# 2. Torch + base Python deps
# ----------------------------------------------------------------------
uv pip install --python "$PY" \
    --index-url https://download.pytorch.org/whl/cu132 \
    torch==2.12.0 torchvision==0.27.0 triton==3.7.0

uv pip install --python "$PY" \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh \
    transformers==4.57.3 timm==1.0.22 kornia==0.8.2 zstandard \
    diffusers==0.37.1 accelerate==1.13.0 gradio plyfile==1.1.3 \
    ninja huggingface_hub safetensors einops

# ----------------------------------------------------------------------
# 3. MoGe + utils3d
# Order matters: MoGe git-pins upstream utils3d; the LDYang694 wheel
# overlay adds the `pt` alias for MoGe-v2 plus a `depth_map_to_point_map`
# helper that Pixal3D's pipeline calls.
# ----------------------------------------------------------------------
uv pip install --python "$PY" "git+https://github.com/microsoft/MoGe.git"
uv pip install --python "$PY" --reinstall-package utils3d \
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# ----------------------------------------------------------------------
# 4. CUDA extensions
# Set TORCH_CUDA_ARCH_LIST for non-Ada GPUs (e.g. "8.0" A100, "9.0" H100,
# "12.0" Blackwell). Unset = build every arch, which is slow.
# ----------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

WHEEL_DIR="$REPO_ROOT/internal/wheels"

install_or_build () {
    local pkg="$1"
    local src_spec="$2"

    shopt -s nullglob
    local matches=("$WHEEL_DIR/${pkg}-"*.whl)
    shopt -u nullglob

    if [ ${#matches[@]} -gt 0 ]; then
        echo "[install] $pkg from cached wheel: ${matches[0]}"
        uv pip install --python "$PY" "${matches[0]}"
    else
        echo "[build]   $pkg from source: $src_spec"
        uv pip install --python "$PY" --no-build-isolation "$src_spec"
    fi
}

install_or_build natten            "natten==0.21.0"
install_or_build cumesh            "git+https://github.com/JeffreyXiang/CuMesh.git"
install_or_build flex_gemm         "git+https://github.com/JeffreyXiang/FlexGEMM.git"
install_or_build nvdiffrast        "git+https://github.com/JeffreyXiang/nvdiffrast.git"
install_or_build o_voxel           "git+https://github.com/microsoft/TRELLIS.2.git#subdirectory=o-voxel"
install_or_build nvdiffrec_render  "git+https://github.com/JeffreyXiang/nvdiffrec.git@renderutils"

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
