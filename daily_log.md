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

## 2025-10-20

### Repository Setup for Reproducibility
- Created comprehensive `Makefile` with setup/run/visualize commands
- Generated `requirements.txt` from working conda environment
- Created `environment.yml` for conda users
- Set up dual support for both pip/venv and conda environments

### Dependency Management Improvements
- Generated `requirements.txt` using `pip freeze` from working k2vae environment
- Cleaned up conda-specific packages (mkl-service, mkl, etc.) that aren't pip-compatible
- Successfully tested `requirements.txt` in fresh virtual environment
- Verified `pip install -e .` works correctly for editable package installation

### Makefile Implementation
- Added `make setup-pip` for pip/venv users
- Added `make setup-conda` for conda users
- Added `make run` for quick test runs
- Added `make run-k2vae` for full K²VAE training
- Added `make visualize` for results visualization
- Added `make test` for installation verification
- Added `make clean` for cleanup

### Testing and Verification
- ✅ Successfully tested `requirements.txt` in fresh `test_env`
- ✅ Verified `import probts` works correctly
- ✅ Confirmed `pip install -e .` creates proper editable installation
- ✅ Repository now supports both pip and conda workflows independently

### Next Steps
- Test full pipeline with `make setup-pip` and `make run`
- Generate `environment.yml` from conda environment
- Test conda workflow with `make setup-conda`
- Update README with clear setup instructions
