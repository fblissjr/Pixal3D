# Pixal3D — repo context

Last updated: 2026-05-17

This file is a working set of notes for AI assistants and contributors
hitting the install path on hardware that diverges from the official
TRELLIS.2 stack. The repo itself is the SIGGRAPH 2026 Pixal3D codebase,
built on Microsoft's TRELLIS.2 and Direct3D-S2.

## Stack divergence

Upstream (TRELLIS.2 `setup.sh`) targets:

  Python 3.10 + torch 2.6.0 + CUDA 12.4 + cp310 prebuilt wheels

`requirements-hfdemo.txt` lists prebuilt wheels for exactly that stack
(`cp310-cp310-linux_x86_64`, torch 2.6 + cu124). They will refuse to
install on any other Python ABI.

If you can use that combination, do so — `pip install -r
requirements-hfdemo.txt` is fast and well-tested.

`install.sh` at the repo root is for the alternate path: Python 3.13 +
torch 2.12 + CUDA 13.x + sm_89 (RTX 4090). It rebuilds every CUDA
extension from source. Expect ~10–15 min of nvcc work on first run.

## Source of every native dependency

These are the components that must be compiled (or pulled in as prebuilt
wheels) and where their source actually lives. Audit before trusting.

| Package | Source repo | Notes |
|---|---|---|
| `natten` | PyPI (SHI-Labs) | Neighborhood attention. Pin to `0.21.0`. |
| `nvdiffrast` | `JeffreyXiang/nvdiffrast` | Fork of NVlabs/nvdiffrast. Runtime-compiled CUDA kernels. |
| `cumesh` | `JeffreyXiang/CuMesh` | Mesh utilities; CUDA kernels. |
| `flex_gemm` | `JeffreyXiang/FlexGEMM` | Sparse-conv backend; CUDA kernels. |
| `o_voxel` | `microsoft/TRELLIS.2` (subdir `o-voxel/`) | O-Voxel decoder; CUDA kernels. |
| `nvdiffrec_render` | `JeffreyXiang/nvdiffrec` branch `renderutils` | Differentiable PBR rendering. |
| `utils3d` | `EasternJournalist/utils3d` (rebuilt by LDYang694) | Pure-Python wheel. Adds a `pt` alias for MoGe-v2 + a `depth_map_to_point_map` helper vs upstream 0.0.2. |
| `MoGe` | `microsoft/MoGe` | Camera/FOV estimation; pure Python. |

JeffreyXiang is Jianfeng Xiang (Microsoft, TRELLIS / TRELLIS.2 author).
The Storages/Releases hosting is convenience — the underlying source is
in the named repos above.

## Hardware notes

The TRELLIS.2-bundled `flash_attn 3` wheel is sm_90 only. On Ada
(RTX 4090, sm_89) and older it will fail to load. Set
`ATTN_BACKEND=sdpa` to fall back to PyTorch's SDPA kernel.

`TORCH_CUDA_ARCH_LIST` should be set to your GPU's compute capability
before building from source — e.g. `8.9` for Ada, `8.0` for A100,
`9.0` for H100, `12.0` for Blackwell. Otherwise the build targets every
arch in torch's default list and is much slower.

VRAM ceilings observed (RTX 4090 / 24 GB):
- `--resolution 1024 --low_vram` → fits comfortably
- `--resolution 1536` → tight in low-vram mode, OOM-prone in standard mode
- `--resolution 1536` standard mode → designed for H100 / 80 GB

## Gated model fallback

`pipeline.json` references `briaai/RMBG-2.0` for background removal.
That repo is gated on HuggingFace and requires manual access approval.

The underlying architecture (`ZhengPeng7/BiRefNet`) is open and produces
near-identical results — RMBG-2.0 is a BRIA fine-tune of it. To use it
without requesting access, edit the local copy of `pipeline.json`:

  "model_name": "briaai/RMBG-2.0"  →  "model_name": "ZhengPeng7/BiRefNet"

`pixal3d/pipelines/rembg/BiRefNet.py` already defaults to BiRefNet, so
no code changes are needed.

## Repo conventions

- `internal/` is gitignored; use it for local notes, wheel caches, or
  session logs that should not ship with the repo.
- Prebuilt wheels for the Python-3.13 / torch-2.12 toolchain live in
  `internal/wheels/` when present, and `install.sh` will prefer them
  over rebuilding.
