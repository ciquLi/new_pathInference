# 正式启动与运行支撑

这里的Python脚本是运行支撑代码，不是单元测试。正式算法实现位于 `src/pathInference/`，自动化测试位于 `tests/`，独立核验和数据工具已移到 `tools/`。

| 文件 | 用途 |
|---|---|
| `run_experiment.sh` | 唯一正式日期入口，包含PATS运行参数 |
| `run_pats_dates.sh` | 顺序运行0201、0301、0401 |
| `run_registered_experiment.py` | 正式入口桥接，核验恢复和缓存边界 |
| `build_training_cases.py` | 上游训练用例的进程和内存调度 |
| `precompute_downstream_candidates.py` | 供正式入口使用的候选缓存准备 |
| `repair_downstream_inputs.py` | 经原审计约束的旧输入恢复 |
| `normalize_neighbor_protocol.py` | 核验模型接口协议元数据，供入口恢复使用 |
| `run_priority_parallel.py` | 资源优先级调度器，调用正式入口 |
| `run_all_experiments.sh` | 上述优先级调度器的薄入口；当前日期顺序入口为 `run_pats_dates.sh` |

这些脚本之间、以及正式数值代码与调度器之间存在调用关系，因此保留原路径；不为整理新增另一套启动脚本。完整导航见根目录 [DIRECTORY_MAP.md](../DIRECTORY_MAP.md)。
