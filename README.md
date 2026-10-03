# STAMP and HF multi-attribute correlation TOPSIS

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
