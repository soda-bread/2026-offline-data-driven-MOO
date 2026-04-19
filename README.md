# 2026-offline-data-driven-MOO

该仓库用于 **离线数据驱动的多目标优化（MOO）** 实验，核心优化器为 **NSGA-II**，并支持多种代理模型：

- **GPR (RBF)**
- **GPR (Matern)**
- **AutoGluon Quantile Regression (QR)**
- **BNN Ensemble（用集成神经网络近似不确定性）**

---

## Notebook 列表

实验与测试 Notebook（当前仓库内）：

- `Exp1 GPR (RBF).ipynb`
- `Exp2 GPR (Matern).ipynb`
- `Exp3 Autogluon_QR.ipynb`
- `Exp4 BNN.ipynb`
- `[Test] Autogluon_QR.ipynb`
- `[Test] BNN.ipynb`

这些 notebook 都采用一致流程：
1. 安装/检查依赖；
2. 从 `src/` 导入功能模块；
3. 数据生成、代理模型训练、覆盖率/不确定性分析；
4. 使用 `run_experiment` 执行 NSGA-II 优化（标准 survival + dual-ranking）。

---

## 项目结构

```text
.
├── Exp1 GPR (RBF).ipynb
├── Exp2 GPR (Matern).ipynb
├── Exp3 Autogluon_QR.ipynb
├── Exp4 BNN.ipynb
├── [Test] Autogluon_QR.ipynb
├── [Test] BNN.ipynb
├── README.md
└── src
    ├── data.py            # 训练/验证/测试数据生成
    ├── experiment.py      # NSGA-II 实验主循环（指标统计）
    ├── metrics.py         # HV / IGD+ 配置
    ├── models.py          # GPR / AutoGluon-QR / BNN 模型封装与预测接口
    ├── opt_problem.py     # pymoo Problem 封装（支持 GPR/QR/BNN surrogate）
    ├── other_functions.py # 统计等辅助函数
    ├── plotting.py        # 绘图函数（Pareto、z-score 等）
    ├── survival.py        # 标准与 dual-ranking survival
    └── uncertainty.py     # coverage 与 alpha 搜索
```

---

## 环境与依赖

推荐 Python 版本：**3.10+**

主要依赖：

- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- pymoo
- GPy
- autogluon.tabular
- ipython
- jupyter

可使用如下命令安装：

```bash
python -m pip install numpy pandas matplotlib plotly scikit-learn pymoo GPy autogluon.tabular ipython jupyter
```

> 提示：Notebook 的第一个代码块也会自动检查并安装核心依赖。

---

## 快速开始

### 1) 运行 Notebook

在仓库根目录启动 Jupyter，然后打开任意实验 notebook（例如 `Exp1 GPR (RBF).ipynb`）顺序运行即可。

Notebook 会自动将仓库根路径加入 `sys.path`：

```python
from pathlib import Path
import sys
repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
```

### 2) 复用 `src/` 模块

你可以在脚本中直接复用统一接口，例如：

```python
from src.opt_problem import build_problem
from src.data import generate_data
from src.experiment import run_experiment
from src.survival import Survival_standard, Survival_dual_ranking
```

---

## 代理模型与优化接口说明

`src/opt_problem.py` 中 `Benchmark_Problem` 支持以下 `use_surrogate`：

- `"GPR_uncertainty"`：输出 `F` 与 `std`
- `"QR_uncertainty"`：输出 `F` 与 `F_q80/F_q90/F_q95`
- `"BNN_uncertainty"`：输出 `F` 与 `std`

`src/survival.py` 中 `Survival_dual_ranking` 支持两种模式：

- **GPR/BNN**：基于 `F + alpha * std` 进行 hybrid 排序；
- **QR**：通过 `alpha=0.8/0.9/0.95` 选择对应分位数（`F_q80/F_q90/F_q95`）进行 hybrid 排序。

---

## 常用检查

可以先做语法检查：

```bash
python -m py_compile src/models.py src/opt_problem.py src/survival.py src/uncertainty.py
```

---

## 默认实验参数（Notebook）

多数 notebook 默认参数：

- `problem_name = "dtlz1"`
- `n_var = 10`
- `n_obj = 2`
- `n_gen = 100`
- `pop_size = 100`

若只想快速冒烟测试，建议先降低规模（如 `n_gen=10`, `pop_size=30`）。

---

## 备注

- 为保证分支判断一致，建议 `problem_name` 使用小写（如 `dtlz1`）。
- `GPy` / `autogluon.tabular` 在不同平台安装时间可能较长，建议使用干净虚拟环境。
