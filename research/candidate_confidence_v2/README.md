# 路径生成后的置信度评估研究

研究边界：公开证据、邻居模型及选择结果、递归前缀树、增强路径库、训练/验证划分、候选生成器及实际生成后的候选全部冻结。仅改变最终候选打分与选择。第一阶段覆盖 20260201 全部十个 AS，最多 2 CPU，不使用 GPU。

第一组受控实验比较同一 LightGBM、相同 255 维特征及连续 Jaccard 弱监督标签下的两种目标：已有平方误差回归，以及查询内 soft-label 交叉熵排序。参考模型复用已经审计的 v1 模型；新模型继承其树容量、学习率、随机种子等参数，只改变目标。两组均以 pair-disjoint 验证查询的 top1 标签 Jaccard 选择轮次，最多 500 轮，40 轮无提升停止。

排序目标：对每个查询，以 `softmax(Jaccard / 0.1)` 作为固定目标分布，以 `softmax(model_score)` 作为预测分布，最小化交叉熵。每个查询一份损失。温度 0.1 在运行前登记，不根据最终测试分数调整；保留连续标签，不把它们离散化。单候选查询没有排序梯度。模型输出仅用于排序，不声称是经过校准的真实路径概率。

先审计全部十源的候选字节、词表、特征矩阵、标签行序、源内 pair-disjoint 划分及公开证据依赖，再训练全部十源模型。全部模型冻结后预测；全部预测冻结后才开始最终评测。v1 的测试结果已经被研究者看过，因此本实验是已有测试集上的探索性对照，不宣称是全新盲测。若后续提出新配置，必须另登记版本，不能覆盖此版本或按测试成绩回选本轮配置。

实际候选文件通过 SHA256 绑定，直接读取 v1 已保留的文件；不重新生成路径。训练特征缓存直接复用，不重算或复制大数据。预测计算相同的冻结特征，特征不写入大型缓存。逐批提交预测与字节偏移到 SQLite，恢复从最后一次提交继续，不重复前面已完成的预测。写出最终 CSV 后才计算输出哈希，遵守训练/预测期间禁止读取正式标签及历史预测的保护。

产物在 `runs/20260201/candidate_confidence_v2/`：`PROTOCOL.json`、`SNAPSHOT.json`、实时 `PROGRESS.md`、阶段日志和回执、模型、预测、计时、最终 `RESULTS.csv`/`REPORT.md`。失败记录错误并停止，不自动修改参数重试；已完成阶段经哈希校验后复用。

启动或续跑：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B research/candidate_confidence_v2/run.py
```

GMPI 保持结构与输入不变的损失对照已独立登记在 `research/gmpi_confidence_v1/`，与本轮并行调度；之后才考虑新增证据信息或改变网络。不得将 LightGBM 间的目标函数结论直接解释为 GMPI 的结构结论。当前 GMPI 的冻结生产训练代码使用 Smooth L1 回归。

资源约定：当前已冻结的 LightGBM 对照沿用 `rl-aspath-inference` 环境和 2 CPU 配置。用户明确允许使用该 conda 环境或其他合适环境，后续 GMPI 训练及预测使用 GPU；数据准备控制 CPU 数量。每个实验记录实际解释器与依赖版本，已运行配置不在中途切换。此授权不涉及 VS Code 的图形硬件加速。

全部预测冻结、统一评测完成后的独立复核：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B tools/validation/audit_candidate_confidence.py --validate-models --check-candidates --rescore
```

审计检查源内划分、增强库路径集合、模型参数的唯一改动、验证奖励、候选身份与预测成员关系，并独立重算最终指标。全部十源通过后才写 `COMPLETION_AUDIT.json`。与 GMPI 的统一结果汇总入口见其研究目录 README。

两项数值任务运行期间，可单独启动收尾监测器：

```bash
/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B research/candidate_confidence_v2/finish.py
```

监测器每 15 秒更新 `runs/20260201/confidence_comparison/PROGRESS.md` 和 `MONITOR.json`；回执计数不代表进程存活。它等待各实验的全部训练、预测、评测回执与冻结屏障，然后执行上述独立审计；两项审计通过后自动生成统一报告。监测器不会启动或重启数值任务，不修改模型和候选。发现生产任务报错或审计失败会记录错误并退出，互斥锁阻止同时启动多个监测器。当前已有实例运行时无需重复启动。

最后两个源的资源调度可使用 `parallel_predict.py --dispatcher-pid <宿主调度器PID> --worker-pid <当前199524预测PID>`。
它只在十个模型均冻结、前八个预测回执验证通过、原队列正在预测 AS199524 时启用，并拒绝已停止的
调度器或错误 PID/任务。暂挂原调度器后，原 AS199524 计算继续，额外调用原冻结 CLI 预测 AS398465。
两个进程均限定 CPU 6、7，没有增加核心范围，也没有改变特征、模型、候选和数值源码。
AS398465 完成后校验身份并写原格式回执，才恢复原调度器，原队列自动跳过已验证结果并统一评测。
异常/中断时先终止提前启动的子进程、保留已提交的 SQLite 断点，再恢复原队列，防止同源双写。
资源门控期间主调度器显示 T 状态是预期行为；同时核对 `PARALLEL_PREDICTION.json` 中的控制器和
两个预测 PID，以及 `PROGRESS.md` 的逐源落盘记录，不要额外启动第二个资源控制器或重复预测。
