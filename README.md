# STAMP and HF multi-attribute correlation TOPSIS

面向应急场景人才能力评价的 Python 算法复现：STAMP 指标体系、犹豫模糊评价、属性关联测度与 TOPSIS 排序。

**复现范围：算法实现与公开数据核验；完整案例无法凭公开信息独立复算。** 论文仅公开 c1、c2、c25、c26 四列评价值，其他 22 列未公开。项目不使用随机数据冒充原始样本，不将论文排序硬编码成算法输出。

依据：《STAMP 和 HF 多属性关联 TOPSIS 的应急场景人才能力评价》，2026，46(1)：207–219，DOI：10.13800/j.cnki.xakjdxxb.2026.0119。项目不包含作者、个人署名、邮箱或原始 PDF；算法采用不含人名的命名。

## 快速运行

Python 3.10 或以上，在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py audit
python run.py demo
```

`audit` 核验公开表格，输出 JSON 与中文报告。`demo` 固定种子生成完整 4×26 模拟矩阵，输出排序、中间矩阵、风险敏感性曲线和基线比较。所有模拟输出有明确标记。

Windows 指定环境：

```powershell
D:/software/miniconda3/envs/myenv/python.exe -m pip install -r requirements.txt
D:/software/miniconda3/envs/myenv/python.exe run.py demo
```

## 实际数据

复制 `data/paper_full_template.json`，补齐 `null` 单元格，每格是 [0,1] 内非空数值数组，然后运行：

```bash
python run.py evaluate --input data/your_case.json --theta 0 --output results/your_case
```

`matrix` 为方案×指标×不定长隶属度；`singletons` 为单指标模糊测度；`benefit` 为各指标是否效益型的布尔列表；`alternatives` 为唯一方案编号。指标与测度次序必须相同。缺失值与非法输入会被拒绝。

## 项目结构

|文件|功能|
|---|---|
|`stamp_hf/core.py`|风险补齐、成本转换、理想解、距离、贴近度、独立 RMS 与 VIKOR 风格基线|
|`stamp_hf/capacity.py`|λ 求根、子集测度、最大分解积分、二可加模型、精确小规模 LP|
|`run.py`|数据核验、模拟实验、自定义数据评价、CSV/JSON/PNG 导出|
|`data/indicators.json`|26 项指标与三类 STAMP 维度|
|`tests/test_model.py`|数值测试、LP 交叉验证与边界测试|
|`docs/method.md`|公式映射、修正及复现边界|
|`results/`|实际运行得到的核验与模拟实验输出|

STAMP 是定性建模过程，此处提供结构化指标，不声称自动从事故描述生成专家指标。

## 已发现的差异

- 表 6 的单点测度之和为 0.655；归一化求得 λ≈1.2833034423，而非原文 1.2。
- λ=1.2 时全集测度约 0.9717243275，不满足全集测度等于 1。
- 公开数据中 a2 的 c1 正理想解距离为 0，原文列为 0.1205。
- c26 负理想解最后一项应为 0.8，原文列为 0.9。
- 一般 λ 测度与二可加测度不是同一个模型，代码分别实现。

完整案例和表 10 的参数不齐全，基线仅作明确约定的算法对照，不宣称重现原文所有对照方法。详细差异见 `results/audit/paper_audit.md`。

## GitHub

建议仓库名：`stamp-hf-correlation-topsis`。建立空仓库后，将本目录内容上传即可；也可用本地 Git 按仓库页面给出的命令推送。源码不包含个人名称，GitHub 的账户与提交归属由平台管理。
