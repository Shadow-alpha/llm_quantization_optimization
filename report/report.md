# 资源约束下大语言模型量化部署的优化建模与算法分析

## 摘要

大语言模型在推理部署阶段通常面临显存占用高、访存带宽压力大、推理延迟高等问题。量化通过使用低比特整数表示权重、激活或 KV cache，可以在较小精度损失下显著降低存储和计算成本。本文从最优化建模角度讨论资源约束下的大语言模型量化部署问题，将量化参数、每层 bit-width、outlier 处理方式、group size 与硬件约束统一为优化变量，并分析 GPTQ、AWQ、SmoothQuant、LLM.int8、HAQ、HAWQ、Q-BERT 与 ZeroQuant 等方法背后的优化思想。实验部分计划在小型语言模型上实现统一的后训练量化框架，对比不同算法在 perplexity、显存、延迟与 Pareto 前沿上的表现。

**关键词：** 大语言模型；后训练量化；混合精度；Hessian；资源约束优化；Pareto 前沿

## 1. 引言

大语言模型（Large Language Models, LLMs）通常包含数亿到数千亿参数。若使用 FP16 或 BF16 表示，每个参数需要 2 字节存储；在长上下文推理中，KV cache 还会进一步增加显存占用。因此，在有限 GPU 显存、推理延迟和能耗约束下部署 LLM，是一个典型的资源约束优化问题。

量化（quantization）将连续浮点权重或激活映射到离散低比特集合。例如 INT8、INT4 量化可显著减少模型存储和访存成本。后训练量化（Post-Training Quantization, PTQ）无需重新训练完整模型，因此在 LLM 部署中尤为重要。量化并非单纯的数据压缩问题，它需要在精度损失、显存占用、延迟、硬件支持之间进行权衡。因此，本文将 LLM 量化部署抽象为约束优化问题，并分析不同算法如何近似求解该问题。

## 2. 量化问题的基本形式

设某一线性层权重矩阵为

$$
W \in \mathbb{R}^{m \times n}.
$$

对于给定 scale 参数 \(s > 0\)，对称均匀量化可写为

$$
\hat{W}
= Q_s(W)
= s \cdot \mathrm{clip}
\left(
\mathrm{round}\left(\frac{W}{s}\right),
q_{\min}, q_{\max}
\right).
$$

若使用 \(b\) bit 表示，则通常有

$$
q_{\min} = -2^{b-1}, \qquad
q_{\max} = 2^{b-1}-1.
$$

最简单的 scale 选择问题可写为

$$
\min_{s>0}
\left\|W - Q_s(W)\right\|_F^2.
$$

若考虑非对称量化，引入 zero-point \(z\)，则有

$$
\hat{W}
= s \cdot
\left[
\mathrm{clip}
\left(
\mathrm{round}\left(\frac{W}{s}\right)+z,
q_{\min}, q_{\max}
\right)
- z
\right].
$$

此时需要联合选择 \((s,z)\)。进一步地，scale 可采用 per-tensor、per-channel 或 per-group 形式，不同粒度会影响量化误差和存储开销。

## 3. 资源约束下的统一优化建模

设原始模型为 \(f_W\)，量化后模型为 \(f_{Q(W;\theta)}\)，其中 \(\theta\) 表示量化策略，包括：

- 每层 bit-width：\(b_1,\dots,b_L\)；
- scale 与 zero-point；
- per-tensor、per-channel 或 per-group 粒度；
- activation、weight 与 KV cache 是否量化；
- outlier 通道处理方式；
- group size 与硬件 kernel 选择。

给定校准数据集 \(D_{\mathrm{calib}}\)，可将量化部署写为：

$$
\min_{\theta}
\mathbb{E}_{x \sim D_{\mathrm{calib}}}
\left[
d\left(f_W(x), f_{Q(W;\theta)}(x)\right)
\right]
$$

$$
\mathrm{s.t.}\quad
C_{\mathrm{mem}}(\theta) \le M,
\qquad
C_{\mathrm{lat}}(\theta) \le T,
\qquad
C_{\mathrm{energy}}(\theta) \le E,
\qquad
\theta \in \Theta_{\mathrm{hardware}}.
$$

其中 \(d(\cdot,\cdot)\) 可取输出 logits 的均方误差、KL 散度或下游任务 loss 增量；\(C_{\mathrm{mem}}\)、\(C_{\mathrm{lat}}\)、\(C_{\mathrm{energy}}\) 分别表示显存、延迟与能耗代价；\(\Theta_{\mathrm{hardware}}\) 表示硬件支持的量化格式与 kernel 约束。

该问题通常是非凸、离散、组合优化问题，难以精确求解。现有算法往往从局部二阶近似、层敏感度估计、outlier 分离、矩阵重参数化或硬件感知搜索等角度构造近似解。

## 4. GPTQ：基于二阶信息的权重量化

普通权重量化最小化的是

$$
\left\|W-\hat{W}\right\|_F^2,
$$

但在神经网络中真正影响模型输出的是

$$
XW \quad \text{与} \quad X\hat{W},
$$

其中 \(X\) 表示该层输入激活。因此 GPTQ 关注的问题是

$$
\min_{\hat{W}\in\mathcal{Q}}
\left\|XW - X\hat{W}\right\|_F^2.
$$

展开可得

$$
\left\|X(W-\hat{W})\right\|_F^2
=
\mathrm{Tr}
\left[
(W-\hat{W})^\top X^\top X(W-\hat{W})
\right].
$$

令

$$
H = X^\top X,
$$

则问题可写为

$$
\min_{\hat{W}\in\mathcal{Q}}
\mathrm{Tr}
\left[
(W-\hat{W})^\top H(W-\hat{W})
\right].
$$

这里 \(H\) 可视为局部 Hessian 近似。GPTQ 的核心思想是：量化某些权重后，将产生的误差通过 Hessian 逆或近似逆传播到尚未量化的权重上，从而进行误差补偿。它体现了二阶优化信息在量化误差控制中的作用。

本文代码中实现课程项目版 GPTQ-like 方法：收集每层输入激活 \(X\)，构造 \(H=X^\top X\) 的对角近似，并在 INT4 权重量化后进行局部误差补偿。

## 5. 混合精度量化与组合优化

不同层对量化误差的敏感度不同。例如 attention projection、embedding 层、LM head 或存在 outlier 的层，通常需要更高精度。设第 \(l\) 层 bit-width 为

$$
b_l \in \mathcal{B}=\{2,3,4,8,16\}.
$$

混合精度量化可建模为：

$$
\min_{b_1,\dots,b_L}
\Delta \mathrm{Acc}(b_1,\dots,b_L)
$$

$$
\mathrm{s.t.}\quad
\sum_{l=1}^{L}
\mathrm{Mem}_l(b_l)
\le M,
\qquad
b_l \in \mathcal{B}.
$$

若每层有 \(K\) 种 bit-width 选择，则搜索空间大小为 \(K^L\)，随层数指数增长。HAQ 使用强化学习搜索每层 bit-width；HAWQ 与 HAWQ-V2 使用 Hessian 或 Hessian trace 衡量层敏感度；Q-BERT 将 Hessian-based mixed-precision quantization 应用于 Transformer 模型。

在本文实验中，将 HAQ、HAWQ、Q-BERT 的共同思想统一为“敏感度驱动的资源分配问题”。设第 \(l\) 层敏感度为 \(S_l\)，第 \(l\) 层使用 \(b\) bit 时的量化误差估计为 \(e_l(b)\)，则有：

$$
\min_{b_1,\dots,b_L}
\sum_{l=1}^{L}
S_l e_l(b_l)
$$

$$
\mathrm{s.t.}\quad
\sum_{l=1}^{L}
\mathrm{Mem}_l(b_l) \le M.
$$

该问题是多重选择背包问题。代码中提供两类求解方式：贪心算法与动态规划算法。

## 6. Hessian 敏感度分析

设量化扰动为

$$
\Delta W = \hat{W} - W.
$$

在一阶项较小的假设下，loss 增量可由二阶泰勒展开近似：

$$
\Delta \mathcal{L}
\approx
\frac{1}{2}
\Delta W^\top H \Delta W,
$$

其中

$$
H = \nabla_W^2 \mathcal{L}(W)
$$

为 Hessian 矩阵。对第 \(l\) 层有

$$
\Delta \mathcal{L}_l
\approx
\frac{1}{2}
\Delta W_l^\top H_l \Delta W_l.
$$

若 \(H_l\) 的特征值较大，则说明该层 loss 曲率较高，量化扰动会被放大，因此应分配更高 bit-width。实际 LLM 中直接计算 Hessian 代价很高，因此常用对角 Hessian、Hessian trace 或激活平方均值作为近似。

本文代码采用两类敏感度估计：

$$
S_l^{\mathrm{act}}
=
\mathbb{E}
\left[
\|X_l\|_2^2
\right],
$$

以及单层量化误差近似：

$$
S_l^{\mathrm{mse}}
=
\left\|W_l - Q(W_l)\right\|_F^2.
$$

前者对应 Hessian trace proxy，后者对应权重扰动大小。

## 7. Outlier 处理：LLM.int8、SmoothQuant 与 AWQ

LLM 权重和激活中常存在 outlier。低比特量化时，少量极端值会拉大量化区间，使大多数普通值的分辨率下降。因此 outlier 处理是 LLM 量化中的关键问题。

### 7.1 LLM.int8

LLM.int8 的思想是将普通通道使用 INT8 计算，而将 outlier 通道保留较高精度。设 outlier 通道集合为 \(S\)，则可写为：

$$
\min_{S}
\mathrm{Err}(S)
\qquad
\mathrm{s.t.}
\quad |S|\le k.
$$

实际实现中可根据 activation 最大值或绝对值阈值确定 \(S\)。本文代码中的 LLM.int8-like 方法检测 activation outlier，将普通通道量化为 INT8，outlier 通道保留原精度。

### 7.2 SmoothQuant

SmoothQuant 使用矩阵重参数化：

$$
Y = XW = (XS)(S^{-1}W),
$$

其中 \(S\) 为对角缩放矩阵。该变换不改变浮点计算结果，但可将 activation 中的 outlier 平滑转移到 weight 中，使 activation 和 weight 都更易量化。对应优化问题为：

$$
\min_S
\mathrm{Err}
\left(
Q(XS), Q(S^{-1}W)
\right).
$$

本文代码实现 SmoothQuant-like 权重侧近似，根据 activation max 与 weight max 构造缩放系数，并观察 W8A8 量化误差变化。

### 7.3 AWQ

AWQ 认为并非所有权重同等重要，activation 较大的通道对输出影响更显著。因此 AWQ 根据 activation 统计量选择重要通道或进行缩放保护。其核心目标可以写为：

$$
\min_s
\left\|
XW - XQ(W \odot s) \odot s^{-1}
\right\|_F^2.
$$

本文代码中的 AWQ-like 方法使用 activation mean 构造 channel scaling，对重要通道进行量化保护。

## 8. ZeroQuant 与端到端 PTQ 流程

ZeroQuant 强调面向 Transformer 的高效后训练量化流程，包括权重量化、激活量化、分组量化以及逐层校准。其工程意义在于将多个局部量化步骤组织为端到端部署流程。

本文代码中实现 ZeroQuant-like 的 group-wise weight quantization。设 group size 为 \(g\)，则对每组权重分别选择 scale：

$$
\hat{W}_{:,G}
=
Q_{s_G}
(W_{:,G}),
\qquad
|G|=g.
$$

相比 per-tensor 量化，group-wise 量化可以降低局部范围差异造成的误差，但需要存储更多 scale 参数。

## 9. 多目标优化与 Pareto 前沿

LLM 部署不仅追求最低精度损失，还需要同时考虑显存、延迟和能耗。因此更完整的形式是多目标优化：

$$
\min_{\theta}
\left(
\Delta \mathrm{Acc}(\theta),
C_{\mathrm{mem}}(\theta),
C_{\mathrm{lat}}(\theta),
C_{\mathrm{energy}}(\theta)
\right).
$$

通常不存在一个策略在所有指标上同时最优，因此需要考虑 Pareto 最优。若不存在另一个策略 \(\theta'\) 满足所有目标均不差且至少一个目标更优，则称 \(\theta\) 为 Pareto 最优解。

实验中将收集不同算法在 memory、latency、perplexity 上的结果，并绘制 Pareto 前沿，用于分析在给定部署约束下应选择何种量化策略。

## 10. 算法对比

| 方法 | 优化变量 | 核心思想 | 对应优化观点 |
|---|---|---|---|
| Uniform Quantization | scale、bit-width | 最小化权重重构误差 | 局部离散近似 |
| GPTQ | quantized weight | 使用二阶信息补偿量化误差 | Hessian 加权最小二乘 |
| AWQ | channel scale | 保护 activation 重要通道 | activation-aware scaling |
| SmoothQuant | 平滑矩阵 \(S\) | 将 activation outlier 转移到 weight | 矩阵重参数化 |
| LLM.int8 | outlier 集合 \(S\) | outlier 高精度、普通通道 INT8 | 稀疏异常值分离 |
| HAQ | 每层 bit-width | 硬件感知自动搜索 | 资源约束策略搜索 |
| HAWQ / HAWQ-V2 | 每层 bit-width | Hessian / trace 衡量敏感度 | 二阶敏感度分配 |
| Q-BERT | group-wise bit-width | Transformer 混合精度 | Hessian-based MPQ |
| ZeroQuant | group scale、量化流程 | 分组量化与逐层校准 | 工程化 PTQ |

## 11. 实验设计

### 11.1 模型与数据集

计划使用小型 causal language model 作为实验对象，例如 `distilgpt2`。校准集与测试集使用 WikiText-2。选择小模型的原因是可以在普通计算资源上完成完整 PTQ 流程，同时保留 Transformer 结构和语言模型 perplexity 指标。

### 11.2 实现方法

代码框架包含以下方法：

- FP16 / FP32 baseline；
- Uniform INT8 / INT4；
- GPTQ-like INT4；
- AWQ-like INT4；
- SmoothQuant-like W8A8；
- LLM.int8-like；
- sensitivity-aware mixed precision；
- ZeroQuant-like group-wise quantization。

### 11.3 评价指标

主要评价指标包括：

$$
\mathrm{Perplexity}
=
\exp
\left(
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}_i
\right),
$$

以及模型参数显存估计：

$$
C_{\mathrm{mem}}
=
\sum_{l=1}^{L}
\frac{N_l b_l}{8},
$$

其中 \(N_l\) 为第 \(l\) 层参数量，\(b_l\) 为该层 bit-width。

推理延迟通过固定 batch 和 sequence length 下的平均前向传播时间估计。

## 12. 实验结果

本节后续填入真实运行结果。

### 12.1 总体结果表

| 方法 | bit 设置 | 显存 / MB | 延迟 / ms | Loss | Perplexity | 备注 |
|---|---:|---:|---:|---:|---:|---|
| FP baseline | 16/32 | 待填 | 待填 | 待填 | 待填 | 原始模型 |
| Uniform INT8 | 8 | 待填 | 待填 | 待填 | 待填 | baseline |
| Uniform INT4 | 4 | 待填 | 待填 | 待填 | 待填 | baseline |
| GPTQ-like | 4 | 待填 | 待填 | 待填 | 待填 | 二阶补偿 |
| AWQ-like | 4 | 待填 | 待填 | 待填 | 待填 | activation-aware |
| SmoothQuant-like | W8A8 | 待填 | 待填 | 待填 | 待填 | outlier 平滑 |
| LLM.int8-like | INT8 + FP outlier | 待填 | 待填 | 待填 | 待填 | outlier 分离 |
| Mixed Precision | mixed | 待填 | 待填 | 待填 | 待填 | 背包优化 |
| ZeroQuant-like | group-wise | 待填 | 待填 | 待填 | 待填 | 分组量化 |

### 12.2 图像占位

后续计划生成以下图像：

- `memory_vs_perplexity.png`：显存与 perplexity 权衡；
- `latency_vs_perplexity.png`：延迟与 perplexity 权衡；
- `pareto_frontier.png`：Pareto 前沿；
- `layer_sensitivity.png`：层敏感度分布；
- `mixed_precision_bits.png`：混合精度 bit-width 分配。

## 13. 分析讨论

从优化建模角度看，各类 LLM 量化算法都可理解为对统一资源约束问题的近似求解：

- GPTQ 使用二阶信息改变量化误差度量；
- AWQ 与 SmoothQuant 通过缩放改变 outlier 分布；
- LLM.int8 将 outlier 通道从普通低精度计算中分离出来；
- HAQ、HAWQ 与 Q-BERT 将 bit-width 选择建模为层级资源分配；
- ZeroQuant 则强调可部署的分组量化与逐层校准流程。

因此，LLM 量化并不是单一数值格式替换，而是一个包含连续变量、离散变量和硬件约束的综合优化问题。

## 14. 总结

本文围绕“资源约束下大语言模型量化部署的优化建模与算法分析”展开，建立了显存、延迟、能耗约束下的统一量化优化模型，并从二阶近似、敏感度分析、outlier 处理、混合精度搜索和 Pareto 前沿等角度分析多种代表性算法。后续实验将基于统一代码框架对比不同算法在小型语言模型上的实际效果，从而验证各类优化思想在部署场景中的作用。

## 参考文献

[1] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323, 2022.

[2] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978, 2023.

[3] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438, 2022.

[4] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339, 2022.

[5] Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. HAQ: Hardware-Aware Automated Quantization with Mixed Precision. arXiv:1811.08886, 2018.

[6] Zhen Dong, Zhewei Yao, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. HAWQ: Hessian Aware Quantization of Neural Networks with Mixed-Precision. arXiv:1905.03696, 2019.

[7] Zhen Dong, Zhewei Yao, Yaohui Cai, Daiyaan Arfeen, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks. arXiv:1911.03852, 2019.

[8] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT. arXiv:1909.05840, 2019.

[9] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. arXiv:2206.01861, 2022.

