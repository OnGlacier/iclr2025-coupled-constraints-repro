# M1：Assumptions 1–3 与 Lemma 1–5 的数值验证

这一模块**不训练算法**，而是检查论文理论部分的常数和算子在官方模型上是否真的满足
假设与引理。它是后续 M2–M4 的“理论地基”。

## 运行

```bash
python repro/m1_theory/lemma_checks.py
python repro/m1_theory/lemma_checks.py --models example_er consensus_ring --trials 10
python repro/m1_theory/lemma_checks.py --models vfl_small
```

常用参数：

- `--models`：`example_er`、`example_ring`、`consensus_ring`、`vfl_small`
- `--trials`：Bregman/算子检查的随机试验次数（默认 20）
- `--seed`：默认 307（与 notebook 一致）
- `--output`：JSON 文件名，默认 `lemma_checks.json`

报告写到 `repro/outputs/reports/lemma_checks.json`。

## 检查内容

### Assumptions 1–3

- `f_i` 的强凸/光滑常数：直接对 `hess_F()` 求特征值，与 `model.mu_f`、`model.L_f`
  比较；
- `A_i` 的常数：`bS = (1/n) Σ A_i A_iᵀ` 的最小正/最大特征值，与 `mu_A`、`L_A`
  比较；
- 图矩阵 `W`：对称性、零空间（`W 1 = 0`）、正特征值范围；
- Bregman 散度上/下界。

### Lemma 1（增广目标 `G`）

在 `R^d × L_m^⊥` 上验证

```
G(x,y) = F(x) + (r/2) ||A x + γ W' y - b||²
r = μ_f / (2 L_A),   γ² = (μ_A + L_A) / μ_W'
```

的强凸常数 `μ_G` 与光滑常数 `L_G`，并对梯度做有限差分检查。

### Lemma 2（`B = [A, γW']`）

把 `B` 限制到 `R^d × L_m^⊥` 后，检查最小正奇异值和最大奇异值与

```
μ_B = μ_A / 2
L_B = L_A + (L_A + μ_A) L_W' / μ_W'
```

的关系。

### Lemma 3（Chebyshev 多项式）

在 `[μ, L]` 上检查

```
P_M(t) = 1 - T_l((L+μ-2t)/(L-μ)) / T_l((L+μ)/(L-μ)),
l = ceil(sqrt(L/μ))
```

满足 `P(0)=0`，且把谱压缩到 `[11/15, 19/15]`。

### Lemma 4（`W'` 与 `K`）

- `W'` 对称、`W' 1 = 0`、谱落在 `[μ_W', L_W']`；
- `K` 的奇异值落在 `[11/15, 19/15]`。

### Lemma 5（官方算子实现）

用稠密矩阵构造作为参照，逐项比较：

- `methods.mulW_prime` 与 `W' @ y`；
- `methods.K_Chebyshev` 与 `Kᵀ(Ku - b')`；
- `methods.grad_G` 与显式梯度。

## 预期结果

三个默认模型上所有检查均为 `PASS`。以 `example_er`（Figure 1 模型）为例：

```
kappa_f = 3140.3067
kappa_A = 26.7495
kappa_W(code) = 88.6597      # (lambda_max/lambda_min)^2
sqrt(kappa_W) = 9.4159       # 若按论文式 (6) 的比值约定
mu_G = 3.185e-4, L_G = 3.8579
```

`example_ring`：`kappa_f≈1531.30`、`kappa_A≈20.22`、`kappa_W(code)≈109.67`。

`consensus_ring`（`ConsensusModel(5,10,'ring',condition_number=10000)`）：
`kappa_f≈9990.02`、`kappa_A=4.0`、`kappa_W(code)≈6.85`。

## 文件说明

- `lemma_checks.py`：全部检查的实现；
- 复用官方 `models.py`/`methods.py`/`utils.py`，不修改它们；
- 稠密参照与官方隐式算子的对比保证了 M2 的独立实现可以逐项对齐。
