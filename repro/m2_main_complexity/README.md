# M2：主算法的独立实现与 Theorem 1 复杂度验证

这一模块做两件事：

1. 用**显式、稠密**的方式重新实现论文的 Main 算法（Algorithm 2），与官方
   `methods.Main` 逐步对齐，确认我们理解的更新公式、参数和资源计数完全一致；
2. 在 `κ_f`、`κ_A`、`κ_W` 三个方向上做受控扫描，拟合资源随条件数的
   log–log 斜率，对照 Theorem 1。

## 文件

- `main_impl.py`：显式 Main。把 `W'`、`B=[A, γW']`、`K=sqrt(P(BᵀB))` 都物化为
  稠密矩阵；每步只做矩阵–向量乘法。
- `compare_with_official.py`：同一模型、同一零初值下，官方 `Main` 与
  `my_main` 的完整误差轨迹和资源计数器逐项比较。
- `controlled_model.py`：条件数可控的二次模型。
- `complexity_sweep.py`：三组扫描、斜率拟合、绘图。

## 1. 独立实现与官方对比

```bash
python repro/m2_main_complexity/compare_with_official.py --steps 20
```

默认比较四个模型：`example_er`（Figure 1）、`example_ring`、
`consensus_ring`、`vfl_small`。判定标准是

- `x_err` 轨迹相对误差 ≤ `1e-8`；
- 最终 `x` 相对误差 ≤ `1e-8`；
- `grad_calls`、`mults_A`、`communications` 三个计数器逐元素完全相等。

本机结果（20 步）：

| 模型 | `x_err` 相对误差 | 最终 `x` 相对误差 | 计数器 |
|---|---:|---:|---|
| `example_er` | `7.9e-14` | `1.0e-12` | 完全一致 |
| `example_ring` | `2.6e-13` | `6.3e-13` | 完全一致 |
| `consensus_ring` | `3.4e-15` | `7.7e-14` | 完全一致 |
| `vfl_small` | `1.1e-11` | `1.0e-10` | 完全一致 |

报告：`repro/outputs/reports/m2_compare.json`。这说明 `main_impl.py` 可以作为
论文 Algorithm 2 的可读参照实现。

## 2. 受控条件数模型

`ControlledQuadraticModel(n, m, d, graph='ring', ...)` 构造

```
f_i(x) = 1/2 xᵀ C_iᵀ C_i x - (C_iᵀ d_i)ᵀ x,
C_iᵀ C_i 的特征值 = geomspace(1, κ_f, d) - θ,
A_i = U diag(sqrt(geomspace(1, κ_A, r))) Vᵀ,  r = min(m, d),
```

因此 `μ_A = 1`、`L_A = κ_A`（在 A 的秩空间上），`κ_f = L_f/μ_f` 也由构造精确
控制；`κ_W` 由官方 ring/ER 图构造决定。参考解 `x*` 通过求解 KKT 系统得到。

## 3. 复杂度扫描

```bash
python repro/m2_main_complexity/complexity_sweep.py
python repro/m2_main_complexity/complexity_sweep.py --families kappa_f --max-steps 4000
```

停止准则：`||x^k - x*||² < 1e-8`。记录到该步为止的累计资源，并做 log–log
最小二乘拟合。

| 扫描 | 取值 | 梯度调用斜率 | A/Aᵀ 乘法斜率 | 真实通信斜率 | 官方通信斜率 |
|---|---|---:|---:|---:|---:|
| `κ_f` | 10, 30, 100, 300, 1000 | **0.502** | **0.502** | **0.502** | 0.502 |
| `κ_A` | 1, 3, 10, 30, 100, 300 | 0.046 | **0.452** | 0.452 | 0.452 |
| `κ_W` | ring n = 5, 8, 10, 12, 16, 20 | 0.034 | 0.034 | **0.513** | 0.034 |

解释：

- `κ_f` 方向：迭代数按 `sqrt(κ_f)` 增长，每步资源数固定，所以三种资源斜率
  都是 `1/2`，与 Theorem 1 的 `O(sqrt(κ_f) log(1/ε))` 一致。
- `κ_A` 方向：迭代数几乎不变，但每步的 Chebyshev 次数
  `n_B = ceil(sqrt(L_B/μ_B))` 随 `sqrt(κ_A)` 增长，所以 A 乘法和通信的斜率
  接近 `1/2`；梯度调用斜率接近 0。`0.046` 是小范围有限步的预渐近偏差。
- `κ_W` 方向：迭代数和每步 A 乘法几乎不变（斜率约 0.03），但 `mulW_prime`
  内部的 Chebyshev 次数 `n_W = ceil(sqrt(κ_W))` 增长，所以**真实**通信斜率
  为 `0.513`，与 `O(sqrt(κ_W))` 一致。

### 官方通信计数器的偏差

官方 `methods.Main` 每步记录

```
2 * ceil(sqrt(kappa_W_prime)) * (n_B + 1),
kappa_W_prime = (19/15)² / (11/15)² ≈ 2.98,
```

其中 `ceil(sqrt(kappa_W_prime)) = 2` 是**常数**；它没有乘上 `mulW_prime` 内部的
`n_W = ceil(sqrt(model.kappa_W))`。因此官方计数器随 `κ_W` 的斜率是 0.034，而不是
理论上的 `1/2`。

`main_impl.my_main` 同时返回：

- `communications`：完全复刻官方计数器，保证与官方实现逐项一致；
- `communications_true = 2 * n_W * (n_B + 1)`：真实 W 乘法次数。

M3/M4 画图仍使用官方计数器，以便和论文图直接对照；讨论通信复杂度时应看
`communications_true`。

## 输出

- `repro/outputs/reports/m2_compare.json`
- `repro/outputs/reports/m2_complexity_sweep.json`
- `repro/outputs/reports/m2_complexity_sweep.csv`
- `repro/outputs/figs/m2_sweep_kappa_f.{pdf,png}`
- `repro/outputs/figs/m2_sweep_kappa_A.{pdf,png}`
- `repro/outputs/figs/m2_sweep_kappa_W.{pdf,png}`

## 备注

- 所有操作都作用在 `R^d × L_m^⊥` 上；`K` 用 `BᵀB` 的特征分解构造，
  `b' = K * lstsq(B, b)`。
- `chebyshev_T` 处理了 `|x|>1` 的分支，避免 ring 图较大时特征值数值越界导致
  `arccosh` 产生 NaN。
- 官方 `Model.mu_B/L_B/kappa_B`、`Model.G/grad_G` 依赖从未赋值的 `bB/bW`，
  本模块不调用它们，而是显式构造 `B`、`K` 和梯度。
