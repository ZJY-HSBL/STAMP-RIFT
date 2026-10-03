# 基于 STAMP 的风险自适应犹豫模糊交互 TOPSIS 稳健评价框架

[English](README.md)

本仓库实现一套面向复杂应急场景人才能力评价的综合多属性决策研究框架。研究将系统理论指标构建、犹豫模糊信息建模、稀疏属性交互识别与稳健 TOPSIS 排序统一在同一计算流程中，重点解决指标构建、专家犹豫信息、指标非独立性以及排序不确定性四类问题。

当前研究框架由四个紧密衔接的层次组成：

1. **STAMP 驱动的指标构建**：从安全约束、控制结构、控制行为和反馈机制出发组织能力评价指标；
2. **风险自适应犹豫模糊补齐**：根据每个犹豫模糊评价单元自身的不确定程度和决策者风险偏好动态确定补齐系数；
3. **稀疏 2-可加属性交互容量**：在保持目标 Shapley 重要度的同时识别具有解释意义的互补、替代与冗余关系；
4. **稳健 TOPSIS 排序**：通过 Monte Carlo 不确定性传播获得分数区间、排名接受度、平均排名和两两优胜概率，而不是只输出单一确定性排序。

## 研究贡献

本项目重点围绕以下方法展开：

- 提出**单元级风险自适应 HFE 补齐机制**，将局部犹豫程度与风险偏好直接耦合；
- 构建**稀疏属性交互识别策略**，在控制交互密度的同时保留具有实际解释价值的互补与冗余关系；
- 实现**可扩展的 2-可加 Choquet 聚合机制**，避免高维指标系统下不必要的指数级组合计算；
- 建立**稳健排名分析层**，联合传播隶属度、指标重要度、风险参数和交互结构的不确定性；
- 形成一条从 **STAMP 指标建模 → 犹豫模糊表示 → 交互容量构建 → TOPSIS 评价 → 稳健性分析** 的完整研究链路。

仓库自带的 `data/synthetic_case.json` 为**明确标注的模拟数据**，用于验证实现正确性和演示完整研究流程。开展实际应用时，应替换为自行采集的案例数据，并完整记录专家评价过程与参数来源。

## 核心方法

对于犹豫模糊元素 $h_{ij}$，定义犹豫度：

$$
u_{ij}=4\operatorname{Var}(h_{ij}),\qquad u_{ij}\in[0,1].
$$

设基础风险偏好为 $r_0$，犹豫敏感系数为 $\beta$，则

$$
r_{ij}=\operatorname{clip}(r_0+\beta u_{ij},-1,1),
\qquad
\theta_{ij}=\frac{1-r_{ij}}{2}.
$$

补齐值为

$$
p_{ij}=\theta_{ij}\max(h_{ij})+(1-\theta_{ij})\min(h_{ij}).
$$

因此，不同评价单元不再共用单一固定补齐参数，而是根据自身犹豫程度动态形成风险响应。

对于 2-可加容量，使用 Möbius 系数 $m_i$ 与 $m_{ij}$，聚合形式为：

$$
C_\mu(x)=\sum_i m_i x_i+\sum_{i<j}m_{ij}\min(x_i,x_j).
$$

其中正的 $m_{ij}$ 表示互补关系，负的 $m_{ij}$ 表示替代或冗余关系。代码通过阈值稀疏化与单调性缩放形成可解释的交互结构，并保持目标 Shapley 重要度。

## 快速运行

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py demo
```

评价自己的案例：

```bash
python run.py evaluate \
  --input data/case_template.json \
  --output results/my_case \
  --risk 0.25 \
  --beta 0.5 \
  --interaction-strength 0.15 \
  --interaction-threshold 0.25 \
  --robust 1000
```

Windows：

```powershell
D:/software/miniconda3/envs/myenv/python.exe -m pip install -r requirements.txt
D:/software/miniconda3/envs/myenv/python.exe -m unittest discover -s tests -v
D:/software/miniconda3/envs/myenv/python.exe run.py demo
```

## 目录结构

```text
stamp_hf/
  hesitant.py      犹豫度、风险自适应补齐、数据校验
  capacity.py      λ 容量与 2-可加容量
  interaction.py   关联估计与稀疏交互构造
  core.py          理想解、距离与 TOPSIS 评价
  robustness.py    Monte Carlo 不确定性传播与稳健排名

data/
  synthetic_case.json  明确标记的模拟案例
  case_template.json   实际研究输入模板
experiments/
  exp_ablation.py
  exp_robustness.py
tests/
docs/
results/
```

## 研究验证设计

运行消融实验：

```bash
python experiments/exp_ablation.py
```

运行 1000 次稳健性分析：

```bash
python experiments/exp_robustness.py
```

当前验证框架支持固定补齐与自适应补齐对比、有无属性交互对比、不同聚合模型对比、参数敏感性、排名稳定性、Monte Carlo 稳健性及计算规模分析。

实际应用研究还应同步记录指标体系、专家筛选标准、评价量表、HFE 构造规则、效益型/成本型方向、指标重要度来源、关联结构来源以及稳健性参数设置，使研究过程能够被完整检查与复算。

进一步说明见 [docs/method.md](docs/method.md)、[docs/research_intro_zh.md](docs/research_intro_zh.md) 和 [docs/research_validation.md](docs/research_validation.md)。

## License

MIT。
