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
```

当前 `run_resource_sweep.py` 还是占位入口，后续用于在不同显存预算下重复运行 mixed-precision 实验。
