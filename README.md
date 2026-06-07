# 资源约束下大语言模型量化部署的优化建模与算法分析

本项目用于最优化理论课程期末报告与汇报。当前版本完成了代码框架与报告数学部分，实验结果章节保留占位，后续可直接补充运行结果、表格和图像。

## 目录

```text
code/                  实验代码
report/report.md        Markdown 报告正文
results/tables/         实验表格输出
results/figures/        实验图像输出
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code/main.py --dry-run
```

`--dry-run` 只检查配置与模块是否正常导入，不下载模型。正式实验可运行：

```powershell
python code/main.py --method uniform --bits 8
python code/experiments/run_all_methods.py
python code/experiments/run_resource_sweep.py
```

## 下载模型到本地

如果服务器无法直接访问 HuggingFace，可以先通过镜像下载模型到 `models/`，之后所有实验都从本地路径加载。

```powershell
# 使用 HuggingFace 镜像下载 distilgpt2
python code/download_model.py --model distilgpt2 --mirror https://hf-mirror.com

# 或者手动指定输出目录
python code/download_model.py --model distilgpt2 --output models/distilgpt2 --mirror https://hf-mirror.com
```

下载完成后，用 `--model-path` 指向本地目录。`--local-files-only` 会禁止 transformers 再访问网络。

```powershell
python code/main.py --model-path models/distilgpt2 --local-files-only --method uniform --bits 8
python code/experiments/run_all_methods.py --model-path models/distilgpt2 --local-files-only
python code/experiments/run_resource_sweep.py --model-path models/distilgpt2 --local-files-only
```

也可以在 shell 中统一设置镜像端点：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python code/download_model.py --model distilgpt2 --output models/distilgpt2
```

`models/` 下的实际模型权重会被 `.gitignore` 忽略，不会上传到 GitHub。

注意：实验默认数据集是 WikiText-2，首次运行仍可能需要通过 HuggingFace datasets 下载数据。若数据集也无法直连，可以在运行实验前保留同一个 `HF_ENDPOINT` 环境变量。

## 运行不同量化方法

单个方法统一通过 `code/main.py` 运行，结果会保存到 `results/tables/`。

```powershell
# Uniform weight quantization
python code/main.py --method uniform --bits 8
python code/main.py --method uniform --bits 4

# GPTQ-like diagonal Hessian compensation
python code/main.py --method gptq_like --bits 4

# AWQ-like activation-aware scaling
python code/main.py --method awq_like --bits 4

# SmoothQuant-like smoothing
python code/main.py --method smoothquant_like --bits 8

# LLM.int8-like outlier handling
python code/main.py --method llm_int8_like --bits 8

# Sensitivity-aware mixed precision
python code/main.py --method mixed_precision --bits 4

# ZeroQuant-like group-wise quantization
python code/main.py --method zeroquant_like --bits 8
```

批量运行所有已配置方法：

```powershell
python code/experiments/run_all_methods.py
```

资源约束扫描入口：

```powershell
python code/experiments/run_resource_sweep.py
python code/experiments/run_resource_sweep.py --budget-ratios 0.25,0.35,0.50,0.75,1.00
```

`run_resource_sweep.py` 会在不同 Linear-layer 显存预算下重复运行 sensitivity-aware mixed precision，输出：

- `results/tables/resource_sweep.csv`
- `results/figures/resource_sweep_pareto.png`
