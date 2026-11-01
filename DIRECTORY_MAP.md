# 项目目录导航

这是同一个Git仓库。按文件用途分目录，不再在项目外复制工作区。

| 目录 | 放什么 | 阅读时怎么判断 |
|---|---|---|
| `src/pathveil/` | 正式方法及其运行依赖 | 邻居选择、前缀树、PATS、GMPI、正式评测 |
| `configs/` | 正式方法参数、日期登记 | 决定跑哪个日期、用哪些规则 |
| `scripts/` | 正式启动和运行支撑脚本 | 日常只需 `run_experiment.sh`；其余供调度和恢复使用 |
| `tests/` | 自动化测试 | 文件均以 `test_` 开头，检查代码行为，不是实验方法 |
| `tools/validation/` | 版本核验和迁移回归工具 | 人工执行检查，不参与普通路径推测 |
| `tools/data/` | 数据准备工具 | 检查输入或下载公开数据 |
| `tools/legacy/` | 旧流程维护工具 | 历史兼容用途，不是当前PATS入口 |
| `research/` | 独立研究实验的方案、参数与入口 | 后续两AS路径吸纳实验在这里独立登记，当前尚未实现 |
| `data/` | 大型数据和派生输入 | 数据，不是代码 |
| `models/` | 模型资产 | 权重，不是方法源码 |
| `runs/` | 运行日志、候选、评测结果 | 产物，不作为生产代码来源 |
| `docs/` | 文档和固定版本清单 | 阅读流程、结果和版本身份 |

## 读正式方法的顺序

1. 根目录 [WORKFLOW.md](WORKFLOW.md)：完整算法流程。
2. [scripts/run_experiment.sh](scripts/run_experiment.sh)：启动命令和可调参数。
3. [src/pathveil/pats/policy.py](src/pathveil/pats/policy.py)：21／5及ρ分组规则。
4. [src/pathveil/pats/workflow.py](src/pathveil/pats/workflow.py)：完整执行顺序。
5. `src/pathveil/upstream/`、`prefix_tree/`、`downstream/`、`core/`：各阶段数值实现。

`src/pathveil/experiment/`中的experiment表示正式实验流程的公共基础模块，不是临时试验目录；其中调度、输入准备、训练和评测是正式方法依赖。`evaluation`是计算正式准确率，`tests/test_*.py`才是检查软件行为。

`src/pathveil/rho_adaptive/`等旧数值模块仍为历史结果、回归和checkpoint兼容保留。此次不移动它们，以保留已经绑定的模块路径和代码哈希。

## 版本边界

`pats-v1.0-21x5`保留整理前的完整稳定快照和五日期基线，标签不移动。目录分类后的快照另记为 `pats-v1.0-21x5-layout1`；数值源码、正式日期配置和运行入口与原标签一致。

原始发布清单和50条逐AS结果保持不变。新的布局清单只记录目录调整后的文件身份，并通过原发布清单的哈希引用原模型和结果，不重写旧实验记录。

以后新增研究实验从稳定版本建立独立分支；实验方案与入口归入 `research/<实验名>/`，测试归入 `tests/`，结果归入独立的 `runs/<日期>/<实验名>/`。
