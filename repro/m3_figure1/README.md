# M3：Figure 1（合成线性回归）与增强系数 `r` 消融

对应论文 Figure 1 与 `example.ipynb` 中的两个实验。

## 1. Figure 1

### 配置（与 notebook 完全一致）

```python
np.random.seed(307)
model = ExampleModel(20, 10, 3, 'erdos-renyi',
                     edge_prob=5/20, gossip=False)
NUM_STEPS = 10_000
Tracking-ADMM: 10 * NUM_STEPS
DPMM:          100 * NUM_STEPS
Main:          NUM_STEPS
```

即 `n=20`、`m=10`、`d_i=3`、Erdős–Rényi 图平均度 5。该模型上
`κ_f≈3140.31`、`κ_A≈26.75`、`κ_W(code)≈88.66`。

### 运行

```bash
# 完整实验（约几分钟）
python repro/m3_figure1/reproduce_figure1.py

# 快速检查：Main 100 步，baseline 按比例放大
python repro/m3_figure1/reproduce_figure1.py --smoke

# 只跑部分方法
python repro/m3_figure1/reproduce_figure1.py --methods Main

# 从已保存的 pickle 重绘
python repro/m3_figure1/reproduce_figure1.py --replot
```

`--num-steps` 可覆盖 Main 的迭代数；`--paper-limits` 强制使用论文图的坐标范围
（完整运行时自动启用）。完整运行中 baseline 的迭代数自动按 10×/100× 放大。

### 输出

- `repro/outputs/figs/m3_figure1.{pdf,png}`
- `repro/outputs/dumps/m3_figure1.pkl`
- `repro/outputs/reports/m3_figure1.json`

三个面板分别是 `x_err = ||x^k - x*||²` 对累计梯度调用、A/Aᵀ 乘法、通信次数，
与论文 Figure 1 的坐标和范围一致（`grad_calls ≤ 10⁴`，另外两个 ≤ `4×10⁵`，
纵轴 `10⁻²¹–10³`）。完整运行后 Main 的曲线会下降到约 `1e-21` 以下。

## 2. 增强系数 `r` 消融

同一模型、Main 10 000 步，依次覆盖 notebook 中的

```python
r_values = [1e-7, 1e-6, 2e-5, 1e-3, 1e-1, 4e-1]
```

每个 `r` 都通过设置 `model._r` 覆盖默认值

```
r_default = μ_f / (2 L_A)
```

目标函数、约束和参考解均不变。

### 运行

```bash
python repro/m3_figure1/ablation_r.py
python repro/m3_figure1/ablation_r.py --smoke
python repro/m3_figure1/ablation_r.py --r-values 1e-7 1e-5 1e-3 1e-1
```

### 输出

- `repro/outputs/figs/m3_r_ablation.{pdf,png}`
- `repro/outputs/dumps/m3_r_ablation.pkl`
- `repro/outputs/reports/m3_r_ablation.json`

三个面板与 notebook 相同（梯度调用、A/Aᵀ 乘法、通信），图例在图的右侧。
JSON 中记录每个 `r` 的最终误差、总资源、运行时间、是否达到 `1e-15` 以及是否
出现非有限值。

## 备注

- baseline 的放大倍数直接来自官方 notebook：Tracking-ADMM 10×、DPMM 100×。
  运行完整实验时请预留数分钟。
- 通信面板沿用官方计数器，未计入 `mulW_prime` 内部的 Chebyshev 次数；原因与
  影响见 `repro/m2_main_complexity/README.md`。
- `--smoke` 会关闭论文坐标范围，否则 100 步的曲线在 `10⁴` 的横轴上几乎不可见。
- `ablation_r.py` 直接复用 `reproduce_figure1.build_model`，保证两个实验的模型、
  随机种子和参考解完全一致。
