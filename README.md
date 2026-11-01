# PathVeil

当前稳定方法是 **PATS 21／5**：同日邻居模型 → 按源 AS 的ρ构造完整前缀树 → 拼接训练尾段并由255维路径模型筛选增强库 → PATS生成训练和查询候选 → GMPI训练、选模和独立评测。

数值基线标签为 `pats-v1.0-21x5`，本次目录分类快照为 `pats-v1.0-21x5-layout1`。五个BGP日期（20260201、20260301、20260401、20260501、20260601）各10个AS的已完成21／5结果，按明确的运行编号登记在[版本说明](docs/releases/pats-v1.0-21x5.md)和[机器清单](docs/releases/pats-v1.0-21x5.json)中。版本清单保留历史代码身份；当前代码快照不等于所有历史运行当时的代码。

## 目录用途

| 目录 | 用途 |
|---|---|
| `src/pathveil/` | 正式方法实现：邻居选择、前缀树、PATS、GMPI |
| `configs/`、`scripts/` | 正式参数、日期登记、启动与调度 |
| `tests/` | 自动化软件测试 |
| `tools/` | 独立核验、数据准备和历史维护工具 |
| `research/` | 后续独立研究实验的方案、参数和入口 |
| `data/`、`models/`、`runs/` | 数据、模型、运行产物 |

详细文件导航见根目录 [DIRECTORY_MAP.md](DIRECTORY_MAP.md)。本项目内按用途分类，不在外部复制工作区。

## 运行

唯一正式日期入口：

```bash
bash scripts/run_experiment.sh 20260301
```

参数默认值集中在 [scripts/run_experiment.sh](scripts/run_experiment.sh)，环境变量可以覆盖。方法配置来自 `configs/workflow_pats.json`，由 `src/pathveil/pats/policy.py` 合并运行参数后写入身份。日期必须在 `configs/dates/` 明确登记；不使用相邻日期替代缺失输入。

```bash
# 仅显示计划，不启动训练或候选生成
bash scripts/run_experiment.sh 20260301 --plan-only

# 三个早期日期按顺序运行；同一日期的路径生成最多并行4个AS
bash scripts/run_pats_dates.sh
```

默认环境为 `/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python`，可用 `PATHVEIL_PYTHON` 指定等价环境。依赖版本记录在 `requirements.lock.txt`，不自动升级。GMPI可以使用CUDA，CPU运行可传 `--device cpu`。恢复已有运行及参数说明见[运行说明](docs/running.md)。

## 固定方法

完整流程以根目录 [WORKFLOW.md](WORKFLOW.md) 为准。

| 项目 | ρ≤1 | ρ>1 |
|---|---|---|
| 首跳 | 两个模型各Top3保护，按原综合分和支持度选择，最多21个 | 两个模型各Top3并集，最多6个 |
| 递归门控 | 当前前缀有效训练尾段少于300条才递归 | 所有选中首跳都递归 |
| 递归宽度 | 5，受合法候选数量限制 | 同左 |
| 增强 | 源树＋TRAIN donor tails，原255维模型评分 | 同左 |
| 筛选 | 源池相对分≥0.75，目标最低1000，数量无上限 | 同左 |
| 训练／查询候选 | 同一完整树、同一增强库、同一PATS，容量200 | 同左 |
| 下游 | 正常历史Huber GMPI，源内pair-disjoint验证选模 | 同左 |

ρ只计算源AS。构造前缀最多3个AS，完整路径最多20个AS。合法候选至少1000条时按原评分规则补足目标，并保留截止分数的并列路径；合法候选不足1000时全部保留并记录缺口。0501 AS328840的885条是这一规则的已登记实例。

当前基线的GMPI监督路径来自增强库，未额外混入原始TRAIN中以目标AS开头的路径。训练标签与PATS本地知识必须消费同一入库集合，并核对路径集合哈希。公开证据、训练来源、held-out源和最终路由标签分别管理。测试标签只用于最后评测；unknown关系继续保留，观测支持率不等于转发真实性。

## 输出与核验

```text
runs/YYYYMMDD/pats_rho_v1/<run_id>/
  PROTOCOL.json、RUN_IDENTITY.json         参数、数据、模型、代码身份
  construction/、query_trees/             前缀选择和完整树
  augmentation/as*/                      增强库及入库记录
  T-PG-B/                                候选、GMPI、检查点与评测
  STATUS.json、PROGRESS.md、logs/          运行进度和日志
  REPORT.md、RESULTS.csv、RESULTS.json     逐AS结果
```

`current.json`仅是可变运行指针，可能指向其他参数组；固定基线应按版本清单中的运行编号读取。已完成缓存须核对输入、模型、配置、代码和输出哈希后才可复用，不以重写旧身份跳过检查。

```bash
# 仅读取代码并核对稳定版本；不会启动实验
python -B tools/validation/verify_frozen_release.py

# 同时核验版本清单登记的本地输入、模型、候选和结果
python -B tools/validation/verify_frozen_release.py --assets
```

Git保存代码、配置、测试、文档和版本清单；大数据、模型和运行产物仍留在 `data/`、`models/`、`runs/`，不随代码仓库上传。版本标签不能替代这些本地资产。

## 代码与回归

```text
src/pathveil/pats/          当前PATS流程、入库、候选生成和恢复
src/pathveil/upstream/      同日邻居模型、递归模型和255维路径评分
src/pathveil/prefix_tree/   公开约束、完整树和PATS数值实现
src/pathveil/experiment/    公共准备、GMPI阶段与资源调度
src/pathveil/downstream/    路径库一致性、训练和评测边界
src/pathveil/core/          历史数值实现和checkpoint兼容模块
src/pathveil/data/          日期登记、数据准备和输入映射
src/pathveil/traceroute/    独立traceroute评测
scripts/                   正式薄入口、调度与运行支撑
tests/                     自动化软件测试
tools/                     核验、数据准备及历史维护工具
research/                  独立研究实验的登记与入口
```

旧ρ流程保留作历史回归和缓存兼容，不是当前日期入口的默认方法。

```bash
PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B -m unittest discover -s tests
/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B tools/validation/verify_migration.py --run-dir runs/verification/pats-v1.0-21x5-layout1/march_migration
```

March回归重算冻结特征的模型分数和选择，验证字节一致性，不重新训练或重跑五个日期。固定结果及验证范围见[版本说明](docs/releases/pats-v1.0-21x5.md)。
