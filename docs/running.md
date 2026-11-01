# 运行与恢复

当前方法是PATS 21／5，流程定义见根目录 [WORKFLOW.md](../WORKFLOW.md)，稳定基线运行编号见[版本说明](releases/pats-v1.0-21x5.md)。正式入口：

```bash
bash scripts/run_experiment.sh 20260301
```

Python默认环境为 `/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python`，可用 `PATHVEIL_PYTHON` 指定。参数默认值在 `scripts/run_experiment.sh`，运行时合并到 `src/pathveil/pats/policy.py` 的策略，写入运行身份。21／5、0.75、1000、候选200是稳定版本参数；通过环境变量覆盖属于新的方法配置。

```bash
# 仅读取计划，不生成候选或训练
bash scripts/run_experiment.sh 20260301 --plan-only

# 顺序调度0201、0301、0401
bash scripts/run_pats_dates.sh
```

20260201、20260301、20260401采用各自登记的10源队列；20260501、20260601用AS16276替换AS42473，并单报共同九源。20261015是单源AS63835的独立traceroute评测。输入只按登记日期读取，缺少同日资产时报错。

## 恢复边界

重复相同命令时，调度器核验已完成工件和身份后复用。显式指定已有运行：

```bash
bash scripts/run_experiment.sh 20260301 --run-dir runs/20260301/pats_rho_v1/run_e3f6274a5f3c9ec8
```

此命令属于实际运行授权，不是核验命令。代码或参数身份不一致时不能靠改旧哈希强制续跑。历史运行可能有已登记的迁移或恢复记录，应一起保留；代码版本标签不保证任意旧目录都能按当前身份原地续跑。

`--fresh-run`会创建独立运行并保留旧产物；不要用它代替缓存恢复。`--prepare-only`等分阶段选项会准备或写入对应阶段产物，只有 `--plan-only`用于单纯展示方法。

## 产物和资源

产物位于 `runs/<date>/pats_rho_v1/<run_id>/`。`STATUS.json`、`PROGRESS.md`和 `logs/`记录状态；`RESULTS.json`、`RESULTS.csv`、`REPORT.md`记录逐AS完成结果。`current.json`会随运行变化，稳定基线按版本清单中的明确编号读取。

训练候选默认并行4个AS并共享加载环境。测试候选最多4个隔离源进程，源内单worker。路径生成阶段完成后再调度GMPI；GPU任务上限2，并受显存、内存预算约束，默认保留10GiB内存余量。GMPI可以用CUDA；这些设置与VS Code硬件加速无关。

## 稳定版本核验

```bash
python -B tools/validation/verify_frozen_release.py
python -B tools/validation/verify_frozen_release.py --assets
```

核验工具仅读取文件，不启动、修复或重写实验。第一条检查代码、配置、入口和测试；第二条同时检查版本清单登记的本地资产。Git标签固定代码，数据和模型仍需在本地保留。

March迁移回归命令及验证范围见[版本说明](releases/pats-v1.0-21x5.md)。外部 `asinfer/files`语料只读；原始RIB归档不是已解析数据的训练前置条件。
