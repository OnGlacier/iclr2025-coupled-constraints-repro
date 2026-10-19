# M4：Figure 2（VFL）、consensus 补充实验、标签分布式 VFL

## 1. Figure 2：VFL 线性回归

论文 Section 6 / Figure 2 的配置：

```python
VFL(num_nodes=7, lmbd=1e-2, title='mushrooms',
    sample_size=100, graph='erdos-renyi',
    edge_prob=5/100, gossip=False,
    labels_distribution=False)
```

注意 VFL 的参数名：`num_nodes=7` 是节点数；`sample_size=100` 对应论文 caption
中的 `m=100`；mushrooms 的 112 个特征被均分，因此每个节点 `d=16`，总变量维度
`n*d + m = 212`，约束行数为 `n*m = 700`。

论文只说明使用 Erdős–Rényi 图，没有给平均度。脚本沿用官方 notebook 的 VFL
约定 `edge_prob=average_degree/sample_size=5/100`；`get_ER_W` 会重采样直到图
连通，因此 n=7 时实际生成的是 6 条边、平均度约 1.71 的连通稀疏图（不是平均度
5），对应 `κ_W≈277.2`。实验报告里的 `num_edges`、`actual_average_degree` 和
`degrees` 记录的是实际图，不是目标参数。

迭代设置：

- Main：1 000 步；
- Tracking-ADMM、DPMM：100 × 1 000 步（与 notebook 对慢速 baseline 的放大一致，
  对应论文图横轴约 `10⁵`）。

```bash
python repro/m4_vfl_consensus/reproduce_figure2.py
python repro/m4_vfl_consensus/reproduce_figure2.py --smoke   # 20 步，快检
python repro/m4_vfl_consensus/reproduce_figure2.py --methods Main
```

输出：

- `repro/outputs/figs/m4_figure2.{pdf,png}`
- `repro/outputs/dumps/m4_figure2.pkl`
- `repro/outputs/reports/m4_figure2.json`

三个面板与论文 Figure 2 一致：梯度调用、A/Aᵀ 乘法、通信。完整运行后 Main 的
斜率明显最陡；DPMM 每步内部用共轭梯度，梯度调用计数较高；Tracking-ADMM 与
DPMM 的曲线在相同资源轴上都比 Main 平缓。

## 2. consensus 补充实验

论文把 consensus 优化作为应用与 Appendix A 的归约来讨论，正文没有对应的图；
这一节复现官方 `consensus.ipynb`：

```python
ConsensusModel(5, 10, 'ring', condition_number=10000)
Main:  10_000 步
EXTRA: 300_000 步（30×）
```

```bash
python repro/m4_vfl_consensus/consensus_extra.py
python repro/m4_vfl_consensus/consensus_extra.py --smoke
```

本机完整运行结果：

| 方法 | 迭代数 | 最终 `||x-x*||²` | 梯度调用 | 通信 |
|---|---:|---:|---:|---:|
| Main | 10 000 | `1.33e-27` | 10 000 | 320 000 |
| EXTRA | 300 000 | `9.77e-14` | 300 000 | 300 000 |

Main 在少得多的梯度调用下达到更低误差，但通信次数略高；这与论文强调的
“通信–计算权衡”一致。图见 `repro/outputs/figs/m4_consensus.{pdf,png}`。

## 3. 标签分布式 VFL（notebook extra）

官方 `vfl.ipynb` 的第二个实验，论文正文没有使用；保留它是为了覆盖官方代码的
`labels_distribution=True` 分支：

```python
VFL(2, lmbd=1.0, title='mushrooms', sample_size=1000,
    graph='erdos-renyi', edge_prob=1/1000,
    gossip=False, labels_distribution=True)
Tracking-ADMM / DPMM / Main: 100 步
```

```bash
python repro/m4_vfl_consensus/labels_distributed.py
python repro/m4_vfl_consensus/labels_distributed.py --smoke
```

**APAPC 被省略**：官方 `methods.APAPC` 读取 `model.bB`，`Model.grad_G` 读取
`model.bW`，而 `Model` 从未赋值这两个属性，运行会抛 `AttributeError`。论文
Figure 2 本身只比较 Tracking-ADMM、DPMM、Main，不含 APAPC，因此不影响论文复现。
输出：

- `repro/outputs/figs/m4_labels_distributed.{pdf,png}`
- `repro/outputs/dumps/m4_labels_distributed.pkl`
- `repro/outputs/reports/m4_labels_distributed.json`

## 备注

- 通信面板沿用官方计数器，见 M2 README 中的说明。
- 完整 Figure 2 的 DPMM 每步内部用共轭梯度，100 000 步是主要耗时来源；如果只想
  验证流程，使用 `--smoke`。
- 若 `data/mushrooms.txt` 缺失，VFL 构造会失败；仓库根目录的 `data/` 已包含该文件。
- `consensus.ipynb` 本身有 `average_degree` 未定义、`results` 未初始化两个问题；
  本模块的脚本已显式修正，官方 notebook 未改动。
