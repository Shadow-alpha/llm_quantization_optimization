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

