# K²VAE Setup and Fixes - Daily Log

## 2025-10-17

### Environment Setup
- Created conda environment `k2vae` with Python 3.10
- Installed project dependencies via `pip install .`
- Encountered multiple dependency conflicts

### Dependency Issues Resolved
- Fixed numpy/pandas binary compatibility error
- Resolved PyArrow compilation issues (switched from pip to conda-forge)
- Installed missing `datasets` package
- Downgraded Lightning from 2.6.0.dev0 to 2.1.0 for compatibility
- Fixed PyArrow version conflict (11.0.0 vs 19.0.0)

### Configuration Updates
- Updated `run.sh` with correct dataset and log directory paths
- Added `--trainer.accelerator cpu` for CPU-only execution
- Set `CUDA_VISIBLE_DEVICES=""` to disable GPU detection

### Code Fixes
- Modified `memory_callback.py` to handle CPU-only execution
- Added CUDA availability checks before calling CUDA functions

### Final Status
- ✅ K²VAE training successfully started on ETTh1 dataset
- ✅ Model: PatchTST with 166K parameters
- ✅ Running on CPU with context length 96, prediction length 96

