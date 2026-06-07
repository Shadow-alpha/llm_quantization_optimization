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
python code/download_model.py --model distilgpt2 --output models/distilgpt2 --mirror https://hf-mirror.com
python code/download_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --output data/wikitext --mirror https://hf-mirror.com
python code/main.py --model-path models/distilgpt2 --dataset-path data/wikitext --method uniform --bits 8
```

## 下载模型和数据集到本地

如果服务器无法直接访问 HuggingFace，可以先通过镜像下载模型到 `models/`、数据集到 `data/`，之后实验都从本地路径加载。

```powershell
# 使用 HuggingFace 镜像下载 distilgpt2
python code/download_model.py --model distilgpt2 --mirror https://hf-mirror.com

# 或者手动指定输出目录
python code/download_model.py --model distilgpt2 --output models/distilgpt2 --mirror https://hf-mirror.com
```

下载 WikiText-2 数据集：

```powershell
python code/download_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --mirror https://hf-mirror.com

# 或者手动指定输出目录
python code/download_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --output data/wikitext --mirror https://hf-mirror.com
```

下载完成后，用 `--model-path` 和 `--dataset-path` 指向本地目录。只要路径存在，代码会自动按本地文件加载。

```powershell
python code/main.py --model-path models/distilgpt2 --dataset-path data/wikitext --method uniform --bits 8
python code/experiments/run_all_methods.py --model-path models/distilgpt2 --dataset-path data/wikitext
python code/experiments/run_resource_sweep.py --model-path models/distilgpt2 --dataset-path data/wikitext
```

也可以在 shell 中统一设置镜像端点：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python code/download_model.py --model distilgpt2 --output models/distilgpt2
python code/download_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --output data/wikitext
```

`models/` 和 `data/` 下的实际文件会被 `.gitignore` 忽略，不会上传到 GitHub。

数据集下载脚本不会调用 `datasets.load_dataset` 解析脚本，而是直接用 `huggingface_hub` 下载 dataset repo 中的 parquet 文件；实验读取本地数据时用 pandas/pyarrow 解析 parquet。

## 运行不同量化方法

单个方法统一通过 `code/main.py` 运行，结果会保存到 `results/tables/`。

结果中的 `memory_mb` 是按目标 bit-width 估算的量化存储显存；`actual_tensor_memory_mb` 是当前 fake-quant 实现中 PyTorch 张量的真实占用。`latency_ms` 测量的是当前 PyTorch 前向传播时间，由于代码没有使用真实 INT4/INT8 kernel，它主要用于同一实现下的粗略对比，不代表硬件量化 kernel 的真实加速。

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
