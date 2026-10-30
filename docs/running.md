# 运行与恢复

日常唯一入口：

```bash
bash scripts/run_experiment.sh 20260301
```

当前BGP方法、两类ρ分组和各阶段定义见[日期化流程](rho_workflow.md)，参数集中在`configs/workflow_rho_prefix.json`。
Python默认`/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python`，可显式设置`PATHVEIL_PYTHON`，不自动升级依赖。

## 检查与分阶段准备

```bash
bash scripts/run_experiment.sh 20260301 --plan-only
bash scripts/run_experiment.sh 20260301 --prepare-only
bash scripts/run_experiment.sh 20260301 --upstream-only
bash scripts/run_experiment.sh 20260301 --augmentation-only
```

检查不拟合模型；准备阶段产生独立输入视图；上游阶段核验或训练同日模型；增强阶段冻结完整路径库。普通运行会依次执行完整流程。

当日有效去重原始完整路径数严格大于10000的全部合格源参与4:1:1源划分；正式测试AS独立留出。只有同日、输入/方法/训练源/代码身份均一致的上游缓存可以复用。

## 日期队列

- 20260201、20260301、20260401：各自明确登记的原十源队列。
- 20260501、20260601：AS16276替换AS42473；同时保存共同九源汇总。
- 20261015：已有AS63835 traceroute的独立离线登记流程，不启动新测量。

本次整理不会自动启动下一日期完整实验，也不启动0501、0601训练。它们的日期入口只在显式执行相应命令后运行。

## 产物与资源

当前运行指针：`runs/<date>/rho_prefix_v1/current.json`。其指向的目录中，`STATUS.json`/`PROGRESS.md`记录实际任务，`logs/`保存逐阶段日志，`REPORT.md`/`RESULTS.csv`在每个AS独立评测完成后更新。

CPU上限8个任务、GPU上限2个任务，实际启动受共享内存账本和显存控制；默认保留10 GiB内存。数值线程、内存预算和进程上限都由同一方法配置记录。一次只调度各AS当前就绪的阶段。

重复相同命令时，已完成工件须重新验证哈希，改变输入或配置会生成新的运行目录。需要明确保留旧产物并重新开始时：

```bash
bash scripts/run_experiment.sh 20260301 --fresh-run
```

`--run-dir`可指定本项目`runs/`内的明确目录；身份不一致时拒绝覆盖。失败保留日志、错误和完成产物，中断权重不作为新的初始模型加载。

## 回归

```bash
/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B scripts/verify_migration.py --run-dir runs/20260301/verification/rho_prefix_migration
```

该命令显式重放March冻结特征并比较模型分数和发布CSV，不是重新训练或盲测。当前日期入口不隐式使用March回放替代其他日期。

外部`asinfer/files`语料只读；缺少原始RIB归档不阻断已解析数据的训练。实际用到的公开快照、路由语料、词表、嵌入和模型必须存在且绑定身份。缺少文件时停止并报出具体路径，不选择相邻日期。
