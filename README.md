# PathVeil

20260201、20260301、20260401已固定为`rho_adaptive_full_tree_v2`：同日上游模型 → 按ρ选择邻居及递归宽度 → 原路径模型筛选增强库 → 完整前缀树＋PG候选生成 → 正常历史GMPI → 完整测试集评测。

## 运行

只改日期即可运行已登记的数据集：

```bash
bash scripts/run_experiment.sh 20260301
```

当前固定版本核验并复用登记日期的上游模型、树构造和增强库；缺少同日缓存时显式报错，不用相邻日期代替。三个已登记运行日期均已具备这些资产。

```bash
bash scripts/run_experiment.sh 20260301 --plan-only
bash scripts/run_experiment.sh 20260301 --prepare-only
```

`--plan-only`打印固定规则，不训练模型；`--prepare-only`核验缓存并准备一致的训练/查询输入。完整运行采用同一条日期入口。

默认Python环境为`rl-aspath-inference`，不自动升级依赖。可通过`PATHVEIL_PYTHON`显式指定等价环境。CPU运行可附加`--device cpu`。

## 固定方法

设公开直接邻居数为B₁、商业关系兼容的二级邻居去重数为U₂，ρ＝U₂/B₁²。结构未知时保留unknown，不默认归为低ρ。训练和查询对同一源使用同一棵完整选择树、同一PG、同一增强库和相同候选限制。模型只在候选生成后计算监督标签；不裁掉已选二跳。固定规则如下：

| 阶段 | ρ≤1 | ρ>1 |
|---|---|---|
| 首跳选择 | 原双模型Top3保护、综合排序和0.7平均支持条件，数量上限由32改为21 | 两个原排序模型各取Top3并集，不再补选 |
| 是否递归 | 合法尾段少于300条才递归 | 每个选中首跳都递归 |
| 每根二跳数量 | 原Top10 | `3 + floor(log2(ρ)/2 + 0.5)`，不足时取实际合格数 |
| 增强构造 | 源自身选择树拼接训练尾段 | 同左，展开后只从S-N-A前缀拼接，不能绕过二跳选择 |
| 增强筛选 | 原255列路径模型，源池相对分≥0.75，最低1000，无上限 | 同左，不移植旧邻居池0.90门槛 |
| 训练候选生成 | 完整源树＋PG | 完整源树＋PG |
| 查询候选生成 | 与训练相同的完整源树＋PG | 与训练相同的完整源树＋PG |
| GMPI | 正常历史Huber、源内AS对隔离验证 | 同左 |

21是用户指定的0201原TRAIN K80 P90统计，跨日期固定使用；3是K80中位数对应的起点。ρ公式是固定工程启发式，不宣称理论准确率保证。仅按模型排序选节点，不新增份额预测模型。前缀最多3个AS，完整路径最多20个AS；关系未知或冲突保留其状态。

0.75是完整源候选池内相对模型分，不是转发概率。最低1000规则若使有效阈值放宽，ADMISSION.json单独记录实际阈值；合法候选不足1000时停止该源并记录，不能静默补入不合格路径。

训练标签与PG本地知识消费同一增强库，并核对路径集合哈希。每个日期复用经核验的同日上游模型；不会用其他日期模型替代。每个AS独立重训一个正常历史GMPI，最终参考仅在模型冻结后用于评测。

固定方法配置：[configs/workflow_full_tree_pg.json](configs/workflow_full_tree_pg.json)。日期登记在`configs/dates/`。旧`rho_prefix_v1`结果保留为独立基线；0501、0601及traceroute仍保留各自原登记，此次不启动。

## 输出与恢复

运行目录由日期、方法和代码/配置身份隔离：

```text
runs/日期/rho_adaptive_full_tree_v2/current.json       当前运行目录指针
runs/日期/rho_adaptive_full_tree_v2/运行编号/
  PROTOCOL.json                          日期、参数、上游模型和代码身份
  augmentation/as*/                      各AS增强库及入库规则
  query_trees/                           查询Tree B
  T-PG-B/                                PG输入、训练、选模和独立评测
  STATUS.json、PROGRESS.md                当前调度与进度
  RESULTS.csv、REPORT.md                  每个AS完成后立即更新
models/日期/rho_prefix_v1/运行编号/as*/    最终选中模型
```

CPU任务按AS并行，GPU最多两个任务并行，并共享项目内存准入账本。默认保留10 GiB内存余量。配置变化后创建新运行目录；同一输入的已完成任务逐项校验哈希后复用。未完成的训练检查点不会当作已完成模型，可用`--fresh-run`保留旧产物并重新开始。

## 代码与验证

```text
src/pathveil/experiment/    日期流程、增强、训练、评测和资源调度
src/pathveil/prefix_tree/   公开结构、Tree B和候选生成数值API
src/pathveil/upstream/      原邻居模型、递归构造和255维路径评分
src/pathveil/core/          原PG与GMPI数值实现
src/pathveil/data/          日期登记、数据准备和读取映射
src/pathveil/traceroute/    独立traceroute评测
scripts/                   薄入口、数据工具和迁移检查
```

生产运行不导入`runs/experimental_workspaces/`内的代码。提取的Tree B数值定义保留来源哈希和可执行AST身份，历史实验产物继续保留。

```bash
PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B -m unittest discover -s tests
/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B scripts/verify_migration.py --run-dir runs/20260301/verification/rho_prefix_migration
```

完整说明见[docs/rho_workflow.md](docs/rho_workflow.md)。

## 结果口径

历史0201的85.13%是跨轮次汇总，其中两家低ρ源使用过CE检查点，不能标为新统一入口的已复现结果。新入口对10个AS统一用正常历史Huber，并按内部验证集选择检查点，实际准确率以本次`RESULTS.csv`为准。

Jaccard、完整有序路径匹配、增强库观测支持率分别报告。0201和0301已用于开发；这些结果不证明未见日期泛化或真实转发正确率。最终测试标签只用于冻结模型后的评测，缺候选计零，不删减正式查询。

外部`asinfer/files`路由语料始终只读。原始RIB归档不是运行前置条件；输入为已解析的同日BGP路径。

0201的`rho_top3_v1/experiment_372cfb29912bdbd3`已采用训练/查询均为完整树＋PG，作为相同数值方法的已完成证据保留。旧低ρ原PG/首跳树补跑仅作为历史对照，不再是当前默认。完成源只在完整树、增强库、模型配置、日期和数值后端均核验等价后复用，不按准确率高低选择。
