# HNSW Optimization Makefile

.PHONY: install test quick-test run-example run-experiment clean help

# Default target
help:
	@echo "HNSW优化项目 - 可用目标:"
	@echo "  install       - 安装依赖"
	@echo "  quick-test    - 快速功能验证"
	@echo "  run-example   - 运行使用示例"
	@echo "  run-experiment - 运行完整实验（小数据集）"
	@echo "  run-experiment-large - 运行完整实验（大数据集）"
	@echo "  clean         - 清理生成文件"
	@echo "  help          - 显示帮助信息"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run quick functionality test
quick-test:
	python quick_test.py

# Run example usage
run-example:
	python example_usage.py

# Run small experiment (quick test)
run-experiment:
	python run_experiment.py --n_vectors 1000 --n_queries 100 --dimension 64 --n_modalities 3

# Run large experiment (full evaluation)
run-experiment-large:
	python run_experiment.py --n_vectors 10000 --n_queries 1000 --dimension 128 --n_modalities 5

# Clean generated files
clean:
	rm -rf results/
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Development setup
dev-setup: install
	@echo "开发环境设置成功！"
	@echo "运行 'make quick-test' 验证功能"
	@echo "运行 'make run-example' 查看基本用法"
	@echo "运行 'make run-experiment' 进行快速测试"
