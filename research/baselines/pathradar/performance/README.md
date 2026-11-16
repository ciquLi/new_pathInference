# PathRadar 执行加速适配层

原版 `pathradar_prediction.py` 和日期 `IDENTITY.json` 不变。适配层只改变执行方式：常驻 Linux fork 进程池动态领取目的地；频次表在父进程加载一次；每个工作进程只加载一次六个模型，SQLite 连接各自打开。每 256 个目的地回收一次工作进程，限制私有内存累积。父进程冻结 GC 对象以减少共享大字典的写时复制。邻居排序使用保持首个索引语义的索引表；最短路径遍历只把 AS 字符串列表和元组列表的深拷贝替换为浅拷贝。

不改变训练轮次、阈值、模型、特征、路径展开/剪枝规则或全量目的地。每个目的地仍以临时文件加原子替换发布，完整 `.txt` 缓存继续复用。原始目录和增强方法的正式模块均不修改。

`benchmark.py` 用同一进程中的原版与加速版逐目的地比较完整输出的路径/分数集合、特征序列、最短路径和评估计数。三目的地真实输入核验报告位于 `runs/baselines/pathradar/performance/equivalence_round1_full/REPORT.json`。单任务计时包含逐特征校验开销，且后台其他实验竞争 CPU，不能据此推算整个进程池的提速倍数。动态队列的实际吞吐量应另行观察。

`run_dates.py` 先验证对照报告和原版/加速代码哈希，再调用原有日期运行器；仅将训练集预测的启动命令替换为 `engine.py`。训练、最终全量评估、输入/模型核验与清理仍调用原运行器。每个日期额外保存 `EXECUTION_AMENDMENT_*.json`，保留原身份并记录执行代码、核验报告与工作进程数。

```bash
/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/run_dates.py
```

默认 8 个常驻工作进程，可在启动前以 `PATHRADAR_PREDICTION_WORKERS` 设置数量。已有队列运行时不要重复启动。当前队列 PID、切换记录和日志位于 `runs/baselines/pathradar/performance/{DISPATCHER.json,HANDOVER.json,BATCH.log}`，每个日期的预测日志仍在原来的 `logs/predict_train_*.log`。

可选 C++ 后端见 `native.cpp` 和 `native_engine.py`。它将路径展开、特征提取、候选更新和最短路径遍历移入 CPython C++ 扩展；保留 Python 容器的遍历顺序、原始 XGBoost 调用和 NumPy 概率比较。大文件加载仍在父进程执行一次，再通过 fork 共享给工作进程。此后端目前用于训练集预测，训练和最终评测仍走原流程。

```bash
/home/lxl/anaconda3/bin/python -B research/baselines/pathradar/performance/build_native.py
PATHRADAR_RUN_DIR="$PWD/runs/baselines/pathradar/20260201" \
PATHRADAR_INTERMEDIATE_DIR="$(dirname "$PWD")/pathRadar/20260201_devoid/intermediate" \
PATHRADAR_READONLY_DB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/benchmark_native.py \
  --round 1 --output runs/baselines/pathradar/performance/native_equivalence_round1
```

核验逐目的地比较全部输出路径/分数、完整内部状态，并逐层比较候选顺序、逐候选比较特征。计时的 `native` 分支关闭额外核验开销；`native_checked` 分支仅用于证明一致性。报告通过至少三个真实目的地，且代码、二进制和模型哈希一致后，才允许设置 `PATHRADAR_NATIVE=1` 启动 `run_dates.py`。小规模计时不等同于整个日期的完成时间。

已有 Python 常驻队列需切换时，先运行 `handover_native.py <prediction_pid>` 查看核验和切换计划，再加 `--apply` 执行。该操作保留全部已原子发布的 `.txt` 结果和训练模型；最多重算工作进程当时尚未发布的目的地，不从头重跑整轮。切换记录保存在 `NATIVE_HANDOVER.json`。

`shortest_native.cpp` 是进一步的整数图实现：一次保留原关系集合的迭代顺序，再用 C++ 数组、整数路径和本地状态表执行原 PathRadar 遍历，完成后按原顺序发布路径和关系状态。它保留原算法的关系约束和优先级更新，不能替换为普通 BFS，也不套用候选阶段的九跳限制。打包后的图只读，在父进程构建后通过 fork 共享；XGBoost 的模型、配置和调用保持不变。

`build_shortest.py` 独立构建该扩展，`benchmark_shortest.py --output runs/baselines/pathradar/performance/shortest_equivalence_round1` 比较它与当前 C++ 桥接版本的最短路径状态、完整预测结果和耗时。通过真实输入核验后，可通过 `PATHRADAR_SHORTEST=1` 选择 `shortest_engine.py`；现有 C++ 队列的切换用 `handover_native.py <prediction_pid> --backend integer --apply`，记录写入 `INTEGER_HANDOVER.json`。原先已核验的桥接源码和构建产物保留，便于回退。

候选阶段只读取最短路径长度表，整数图模式直接发布该表；最短路径详细路径、优先级路径和关系状态按需重建。核验仍逐项检查这些详细状态及插入顺序；正式预测无需每次将全部 C++ 辅助状态转回 Python 字典。此处的按需重建不改变最终候选路径或评估输出。

`features_native.cpp` 将静态特征证据转入不可变的 C++ 表：AS 编号映射为整数，节点度和 core 标记存入数组，关系、排名和边频次按整数边查找，前后缀支持在 C++ 字符串表中查询。每个目的地的最短距离编码一次；特征按原列序批量写为连续的 int64 缓冲区，NumPy 用 `frombuffer` 读取后直接交给原 XGBoost `predict_proba`。不更改模型、阈值、预测调用或候选迭代顺序。证据只在父进程构建一次，fork 后共享；代价是初始化和额外的只读表内存。

```bash
/home/lxl/anaconda3/bin/python -B research/baselines/pathradar/performance/build_features.py
# 用与上面相同的 PATHRADAR_RUN_DIR、PATHRADAR_INTERMEDIATE_DIR 和线程环境运行：
/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/benchmark_features.py \
  --output runs/baselines/pathradar/performance/features_equivalence_round1
```

特征核验包含每个候选每一列的值和 dtype，以及完整预测分数、路径、状态和评估计数。`features_checked` 仅用于验证，提速使用关闭逐项核验的 `features` 分支测量。通过至少三个真实目的地及哈希核验后可设置 `PATHRADAR_FEATURES=1` 启动日期队列；从整数图队列切换使用 `handover_native.py <prediction_pid> --backend features --apply`。保留全部已发布结果和模型，切换记录为 `FEATURES_HANDOVER.json`。

`pipeline_engine.py` 继续优化候选展开：将关系集合按原迭代顺序编码为只读整数邻接表，逐层保留模型分组、候选列表、Python 候选集合和稳定排序。候选更新仅缓存不可变分数与阈值的比较，仍调用原 NumPy 标量比较，保留 NaN、阈值相等及重复路径覆盖的语义。构建使用 `build_pipeline.py`；`benchmark_pipeline.py` 对每层路径和分数更新及最终完整状态做核验，并比较最终评估的逐对详情。发布前释放已编码的 Python 频次、排名、关系字典及未使用的 NetworkX 图，避免在 fork 后保留两份证据。

`loader_engine.py` 从 `joinfreq_data.txt` 直接流式构建 C++ 频次表，省去先加载大 Python 字典再复制的过程。其余小型证据保持原解析规则及集合插入顺序；重复频次记录保持最后一条生效。证据完成初始化后才交给候选计算，并由各进程只读共享。`build_loader.py` 独立构建此后端；`benchmark_loader.py` 与原解析器核对**全部**频次、关系、度、core、排名及边频次，并核验真实完整预测和评估结果。

`evaluate_pipeline.py` / `evaluate_loader.py` 复用原 `evaluate_top1_pairs.py` 的查询选择、top-1、并列处理、Jaccard、累积顺序及输出格式。父进程加载一次公开证据，工作进程各加载一次原模型；日期队列最多使用 4 个评估进程，训练集预测仍使用配置的工作进程数。评估阈值仍为 0.7，训练集预测阈值仍为 0.5。XGBoost 的训练、模型、特征列和预测 API 均未更改。

```bash
/home/lxl/anaconda3/bin/python -B research/baselines/pathradar/performance/build_pipeline.py
/home/lxl/anaconda3/bin/python -B research/baselines/pathradar/performance/build_loader.py
# 使用对应日期的 PATHRADAR_RUN_DIR / PATHRADAR_INTERMEDIATE_DIR：
/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/benchmark_pipeline.py \
  --output runs/baselines/pathradar/performance/pipeline_equivalence_round1
/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/benchmark_loader.py \
  --output runs/baselines/pathradar/performance/loader_equivalence_round1
```

`run_dates.py` 默认选择全部已核验加速；启动时要求源文件、模型、构建依赖及二进制哈希一致。若需回退，设置 `PATHRADAR_LOADER=0` 后使用原来的后端开关，例如 `PATHRADAR_PIPELINE=1` 仅选择候选优化与释放重复证据。运行中的特征加速队列可用 `handover_native.py <prediction_pid> --backend loader --apply` 切换：保留所有已原子发布的预测和训练模型，最多重算切换时尚未发布的工作进程任务。记录位于 `LOADER_HANDOVER.json`。不要同时启动第二个日期队列。

最终评估必须启用 `--include-pair-details 1`。日期运行器将逐查询结果保留在 `testing/top1_eval_round2_full.json` 的 `pair_details` 中，并导出 `top1_eval_round2_full.pairs.csv`。CSV 字段为 `src,dst,predicted_path,jaccard,valid_prediction,exact_match`；预测路径用 `->` 连接 AS，无有效预测的查询仍保留，路径为空、有效标记为 0、Jaccard 为 0。JSON 中对应的有效标记为 `covered`。Jaccard 沿用原评估器的 AS 集合定义，多条真实路径取最高分。

`pair_details.py` 检查明细与该日期完整测试查询集合一致、无重复，验证路径端点和有效标记，并复算覆盖数、完整匹配数和 Jaccard 均值；CSV 写入后再次读取核对数量。检查回执 `top1_eval_round2_full.pairs.audit.json` 保存实际记录数、总查询数和 JSON/CSV 哈希，日期 `COMPLETE.json` 也绑定此回执。检查通过后才清理中间证据，明细 JSON、CSV、回执和模型继续保留。

对于已经完成但未保存明细的 20260301，可用 `/home/lxl/anaconda3/bin/python -B -u tools/validation/backfill_pathradar_details.py 20260301` 补评估。该工具验证原日期输入，复用六个第 2 轮模型，仅重建已清理的评估证据，不重训模型、不重算训练阶段的最短路径数据库。完成后检查模型哈希、全量查询集合以及新旧汇总指标；旧汇总另存 `top1_eval_round2_aggregate_before_details.json`。若重建证据的并列次序导致分数变化，将明细及差异报告单独保留，不自动覆盖旧汇总。状态和日志在 `control/pair_details_backfill/` 与 `logs/pair_details_backfill.log`。同一日期有互斥锁，已有任务运行时无需重复启动。

`handover_evaluation.py <dispatcher_pid> <shortest_pid>` 用于让已经导入旧日期运行器的调度进程加载新评估参数：先校验原数值实现，再暂停调度父进程，保留正在运行的最短路子进程；子进程结束后重新加载调度器，复用已完成输出。衔接记录与日志为 `EVALUATION_DETAILS_HANDOVER.json` 和 `.log`，训练集预测仍保持原工作进程数。该工具不修改已冻结的评估、训练或特征实现。

若补评估在 `details_complete_historical_aggregate_differs` 状态停止，说明新明细已完整生成，但分数与历史汇总不同。检查差异后，可在同一补评估命令末尾添加 `--finalize-details`：只核验并整理已生成的结果，不再次预测，单独保存为 `top1_eval_round2_full_reevaluated.json` 及对应 `.pairs.csv`、`.pairs.audit.json`；历史汇总保留。`control/pair_details_backfill/REEVALUATION_COMPLETE.json` 记录新旧差异和产物哈希，校验通过后删除本次临时重建证据。ESR、曲线与对应汇总分数应统一读取这次重评结果，不能将历史汇总数值当作新明细的汇总。原频次文件和运行时顺序未保留，差异原因不能仅凭输入路径数量相同而断定。

0201、0401 补齐队列入口为 `/home/lxl/anaconda3/bin/python -B -u research/baselines/pathradar/performance/backfill_dates.py`。仅运行这两个已登记日期，顺序执行，CPU 亲和性限制在 4、5 两个核心；复用原第 2 轮模型。0201 使用原运行绑定的外部证据缓存，先核对数据库与证据哈希；缓存只读且不清理。0401 只重建已删除的评估证据。队列识别“重评已完成但分数不同”的保护状态，将新结果单独整理，再调用 `tools/validation/audit_pathradar_pair_details.py` 独立检查实际 CSV、日期测试查询集合、输出哈希与汇总指标；通过后才进入下一日期。状态与每 15 秒更新的进度表在 `runs/baselines/pathradar/detail_backfill/`。不要与同一队列重复启动。
