# K²VAE Makefile - Easy setup and execution
# Usage: make <target>


#
# For pip users
# ------------------
# make setup-pip
# make run
# make visualize
#

#
# For conda users
# ------------------
# make setup-conda
# conda activate k2vae
# make run
# make visualize
#

.PHONY: help setup-pip setup-conda run run-k2vae visualize clean test download-datasets

# Default target
help:
	@echo "K²VAE - Available Commands:"
	@echo "  make setup-pip     - Complete pip setup (install + download datasets)"
	@echo "  make setup-conda   - Complete conda setup (create env + install + download datasets)"
	@echo "  make run           - Run K²VAE with PatchTST on ETTh1 (quick test)"
	@echo "  make run-k2vae     - Run K²VAE model on ETTh1"
	@echo "  make visualize     - Generate comprehensive visualizations"
	@echo "  make test          - Run quick test to verify installation"
	@echo "  make clean         - Clean logs and temporary files"
	@echo "  make download-datasets - Download required datasets"

# Complete pip setup (install + download datasets)
setup-pip:
	@echo "🔧 Setting up K²VAE with pip..."
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "📥 Downloading datasets..."
	@if [ ! -d "datasets" ]; then mkdir -p datasets; fi
	python probts/utils/download_datasets.py --data_path ./datasets
	@echo "✅ K²VAE pip setup complete! Ready to run."

# Complete conda setup (create env + install + download datasets)
setup-conda:
	@echo "🔧 Setting up K²VAE with conda..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda not found. Please install Anaconda or Miniconda first."; \
		exit 1; \
	fi
	@echo "📦 Creating conda environment..."
	conda env create -f environment.yml
	@echo "📥 Downloading datasets..."
	@if [ ! -d "datasets" ]; then mkdir -p datasets; fi
	conda activate k2vae && python probts/utils/download_datasets.py --data_path ./datasets
	@echo "✅ K²VAE conda setup complete! Ready to run."
	@echo "💡 Remember to run: conda activate k2vae"

# Download datasets only
download-datasets:
	@echo "📥 Downloading datasets..."
	@if [ ! -d "datasets" ]; then mkdir -p datasets; fi
	python probts/utils/download_datasets.py --data_path ./datasets
	@echo "✅ Datasets downloaded!"

# Quick test run with PatchTST (1 epoch, CPU)
run:
	@echo "🚀 Running K²VAE PatchTST test on ETTh1..."
	@if [ ! -d "logs" ]; then mkdir -p logs; fi
	python run.py --config config/ltsf/etth1/patchtst.yaml --seed_everything 0 \
		--data.data_manager.init_args.path ./datasets \
		--trainer.default_root_dir ./logs \
		--data.data_manager.init_args.dataset etth1 \
		--data.data_manager.init_args.split_val true \
		--trainer.max_epochs 1 \
		--data.data_manager.init_args.context_length 96 \
		--data.data_manager.init_args.prediction_length 96 \
		--trainer.accelerator cpu \
		--model.forecaster.no_training true
	@echo "✅ Test run complete!"

# Run K²VAE model
run-k2vae:
	@echo "🚀 Running K²VAE model on ETTh1..."
	@if [ ! -d "logs" ]; then mkdir -p logs; fi
	python run.py --config config/ltsf/etth1/k2vae.yaml --seed_everything 0 \
		--data.data_manager.init_args.path ./datasets \
		--trainer.default_root_dir ./logs \
		--data.data_manager.init_args.dataset etth1 \
		--data.data_manager.init_args.split_val true \
		--trainer.max_epochs 10 \
		--data.data_manager.init_args.context_length 96 \
		--data.data_manager.init_args.prediction_length 96 \
		--trainer.accelerator cpu
	@echo "✅ K²VAE run complete!"

# Generate visualizations
visualize:
	@echo "📊 Generating comprehensive visualizations..."
	python visualize_results.py
	@echo "✅ Visualizations complete!"

# Quick test to verify installation
test:
	@echo "🧪 Testing K²VAE installation..."
	python -c "import probts; print('✅ K²VAE imports successfully')"
	python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
	python -c "import lightning; print(f'✅ Lightning version: {lightning.__version__}')"
	@echo "✅ All tests passed!"

# Clean logs and temporary files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf probts/__pycache__/
	rm -rf probts/*/__pycache__/
	rm -rf build/
	rm -rf *.egg-info/
	rm -f k2vae_comprehensive_analysis.png
	rm -f k2vae_comprehensive_analysis.svg
	@echo "✅ Cleanup complete!"

# Full pipeline: setup + run + visualize
all: setup-pip run visualize
	@echo "🎉 Complete K²VAE pipeline executed successfully!"

# Full conda pipeline: setup-conda + run + visualize
all-conda: setup-conda run visualize
	@echo "🎉 Complete K²VAE conda pipeline executed successfully!"
