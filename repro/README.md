# 复现模块总览（`repro/`）

本目录把 ICLR 2025 论文 **“Decentralized Optimization with Coupled Constraints”**
的复现拆成四个互相独立的模块。所有新代码都放在 `repro/` 下，官方根目录中的
`models.py`、`methods.py`、`utils.py` 以及三个 notebook **没有做任何修改**。

四个模块与论文的对应关系：

| 模块 | 对应论文内容 | 主要产物 |
|---|---|---|
| M1 | Assumptions 1–3、Lemma 1–5 的数值验证 | `outputs/reports/lemma_checks.json` |
| M2 | 主算法（Algorithm 2）的独立实现、Theorem 1 复杂度斜率 | `main_impl.py` 对比报告、三组 sweep 图 |
| M3 | Figure 1（合成线性回归）与增强系数 `r` 消融 | `m3_figure1.*`、`m3_r_ablation.*` |
| M4 | Figure 2（VFL）、consensus 补充实验、标签分布式 VFL | `m4_figure2.*`、`m4_consensus.*`、`m4_labels_distributed.*` |

所有脚本都从仓库根目录运行，输出统一写到：

```
repro/outputs/
├── figs/       # PDF + PNG
├── dumps/      # 原始结果（pickle，可用于重绘）
└── reports/    # JSON / CSV 汇总
```

## 快速验证（smoke test）

```bash
bash repro/run_smoke.sh
```

约 1–2 分钟，逐个检查四个模块能否跑通，不追求完整曲线。

## 完整复现

```bash
bash repro/run_full.sh
```

会运行 M1、M2 的全部检查，以及 M3/M4 的论文级实验。M3 与 M4 的
baseline 分别有 10×/100× 的迭代放大，完整运行通常需要数分钟到十几分钟；
每一部分也可以单独运行，见各模块 README。

## 环境

使用官方依赖的 conda 环境即可（本机为 `decentralized-opt`）：

```bash
conda activate decentralized-opt
python repro/m1_theory/lemma_checks.py
```

`VFL` 需要仓库根目录下存在 `data/mushrooms.txt`（以及可选的 `a9a.txt`、
`w8a.txt`）。这些文件已经随仓库提供。

## 四个模块的入口

```bash
# M1：Assumptions 1–3 + Lemma 1–5
python repro/m1_theory/lemma_checks.py

# M2：独立 Main 与官方对比 + 复杂度 sweep
python repro/m2_main_complexity/compare_with_official.py --steps 20
python repro/m2_main_complexity/complexity_sweep.py

# M3：Figure 1 + r 消融
python repro/m3_figure1/reproduce_figure1.py
python repro/m3_figure1/ablation_r.py

# M4：Figure 2 + consensus + labels-distributed
python repro/m4_vfl_consensus/reproduce_figure2.py
python repro/m4_vfl_consensus/consensus_extra.py
python repro/m4_vfl_consensus/labels_distributed.py
```

每个脚本都支持 `--smoke`（或 M1/M2 的小规模参数）做快速检查。

## 已知的官方代码问题

这些是官方仓库本身的问题；复现代码通过独立实现或显式配置绕开，**没有修改官方文件**。

1. **`APAPC` 无法运行**：`methods.APAPC` 读取 `model.bB`，而 `Model` 从未赋值
   `bB`；`Model.grad_G`/`mu_B`/`L_B` 还需要从未赋值的 `model.bW`。因此运行
   `APAPC` 会抛 `AttributeError`。论文 Figure 2 只比较 Tracking-ADMM、DPMM 和
   Main，不含 APAPC；labels-distributed notebook 中的 APAPC 曲线在 M4 中被省略。
2. **`consensus.ipynb`**：cell 1 使用未定义的 `average_degree`（即使 `graph='ring'`
   也会求值），cell 2 在赋值前使用 `results`。M4 的 `consensus_extra.py` 已修正。
3. **`vfl.ipynb` 默认参数与论文不一致**：notebook 当前默认 `n=14`、
   `sample_size=8124`、ring 图；论文 Figure 2 是 `n=7`、`m=100`、Erdős–Rényi
   平均度 5。M4 使用论文配置。
4. **通信计数器**：官方 `methods.Main` 每步记录
   `2*ceil(sqrt(kappa_W_prime))*(n_B+1)`，其中 `kappa_W_prime≈2.98` 是常数，
   没有计入 `mulW_prime` 内部 `n_W=ceil(sqrt(model.kappa_W))` 次 W 乘法。
   M2 同时报告 `communications`（官方口径）和 `communications_true`（真实 W 乘法
   次数），并给出两者随 `kappa_W` 的斜率差异。M3/M4 画图沿用官方口径，以便与论文
   图直接对照。
5. **`kappa_W` 的约定**：官方代码中 `mu_W`/`L_W` 存的是 W 特征值的平方，
   `kappa_W=(λ_max/λ_min)^2`。Figure 1 的 `kappa_W≈88.66` 与论文 caption 一致；
   若按式 (6) 的 `λ_max/λ_min` 理解则应为 `sqrt(88.66)≈9.42`。M1/M2 的报告里
   两个量都给出。

## 目录结构

```
repro/
├── common.py                  # 路径、绘图、JSON/pickle 工具
├── m1_theory/
│   └── lemma_checks.py
├── m2_main_complexity/
│   ├── main_impl.py           # 显式/稠密版 Main
│   ├── controlled_model.py    # 可控 kappa_f, kappa_A 的二次模型
│   ├── compare_with_official.py
│   └── complexity_sweep.py
├── m3_figure1/
│   ├── reproduce_figure1.py
│   └── ablation_r.py
├── m4_vfl_consensus/
│   ├── reproduce_figure2.py
│   ├── consensus_extra.py
│   └── labels_distributed.py
├── run_smoke.sh
└── run_full.sh
```
