# GMPI 候选置信度：仅改变训练损失

本阶段属于路径生成后的独立研究，首先在 20260201 全部十源运行。公开证据、邻居模型与选择结果、递归前缀树、增强路径库、训练/验证划分、实际生成的候选及历史路径输入全部冻结。绑定已保留的 candidate_scorer_v1 实际候选与 candidate_confidence_v2/SNAPSHOT.json；不重新生成路径，不重算增强库。

动机来自代码核对：当前 GMPI 的最终输出对候选维度做 softmax，各候选分数之和为 1；原损失却将这些分数与逐候选的连续 Jaccard 直接做 Smooth L1 回归。多个候选的 Jaccard 可以同时很高，而预测分数受到总和约束。该现象是实验动机，不是对真实测试差距的因果结论。

目标的理论性质：冻结训练器使用 `F.smooth_l1_loss(pathScore, similarity.detach())`，实际 Torch 1.8.1 默认 `beta=1`。对于取值在 `[0,1]` 的概率和 Jaccard 标签，其损失等于平方误差的一半。若暂时把一个查询的输出视为任意有效候选概率向量，最小化该误差相当于把标签向量投影到概率单纯形：`p_i = max(J_i - tau, 0)`，其中 `tau` 使概率总和为 1。这个理想解保留最大 Jaccard 标签的候选；交叉熵的理想解 `softmax(J / 0.1)` 也保留相同的最大标签候选。

例如标签为 `[0.9, 0.8, 0.7]` 时，平方误差的单纯形投影为 `[0.4333, 0.3333, 0.2333]`，本轮交叉熵目标约为 `[0.6652, 0.2447, 0.0900]`，两者都选择第一条。区别在于概率分配和训练梯度，以及有限网络容量、弱标签误差和实际输入语义下的学习效果。标签总和超过 1 本身不足以解释选路准确率差距；目标函数的收益和适用源需要由本轮固定输入的实证对照判断。

两组均使用相同 GMPI 结构、嵌入初值、可训练参数、候选和历史、Adam、batch、学习率、随机种子、训练上限、早停规则。读取历史冻结配置继承参数。新对照从冷初始化开始，两组每个源的初始 state_dict 哈希必须相同；不能只拿一组历史训练结果声称严格控制了运行环境和初始化。

- smooth_l1：保留原生产训练损失，包括其现有 padding 语义。
- query_cross_entropy：对有效候选标签取 `softmax(Jaccard / 0.1)`，以此为目标分布；直接使用网络已经归一化的输出计算 `-sum(target * log(prediction))`，按查询平均。不能再次对网络概率输出做 softmax。padding 只在新目标中屏蔽，是新损失定义的一部分；单候选查询零损失、零排序梯度。

两组均以相同源内 pair-disjoint 验证查询的 top1 弱标签 Jaccard 选检查点；全部模型冻结后预测，全部预测冻结后评测。不得根据历史测试表现为某个 AS 单独调整温度或损失。旧测试结果已知，所以本阶段仍是探索性研究，不宣称是全新盲测。

使用 GPU 训练和预测；最多一个 GPU 工作进程、2 个 CPU 线程，按源和变体顺序调度。宿主环境已验证 RTX 3090、`rl-aspath-inference` 的 Torch 1.8.1+cu111 与 CUDA 正常；隔离工具环境中的 cuda_available=False 不能被当作宿主显卡故障。禁止为获得 GPU 而修改 VS Code 的图形加速设置。

实现以独立研究适配器替换生产 trainer.train 中唯一的损失调用，保留其前向传播、优化器更新、数据加载和验证代码；不写入生产模块。替换器必须拒绝非预期源码布局。损失与替换器位于 src/pathInference/research/gmpi_confidence_loss.py，并通过 padding、梯度、单候选及替换范围测试。

启动或续跑：`/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B -u research/gmpi_confidence_v1/run.py`。需要宿主 CUDA 可见的执行环境。实时进度：`runs/20260201/gmpi_confidence_v1/PROGRESS.md`。LightGBM 目标函数对照独立继续运行；两轮结果分别发表。

同一工作进程内复用环境和初始嵌入，避免十源及两种损失反复加载大文件。每个 epoch 原子保存模型、优化器、随机状态和验证历史；中断后最多重做当前未保存的 epoch。模型冻结后删除冗余优化器断点，保留选定模型和训练历史。预测逐批事务提交，断点保存实际候选文件的字节偏移。

完成后的独立复核：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B tools/validation/audit_gmpi_confidence.py --check-weights --check-candidates --rescore
```

该审计逐源验证配对初始化、模型与输入身份、验证检查点选择、候选成员关系和独立测试重评分。候选槽位上限若实际删掉路径，必须另外计算 GMPI 可用候选的 oracle，不能直接套用完整候选库的 oracle。审计输出为 `COMPLETION_AUDIT.json`；任务未完成时仅写 `PRELIMINARY_AUDIT.json`。

LightGBM 与 GMPI 两轮均完成并通过独立审计后，运行 `tools/validation/report_candidate_confidence.py`，产出 `runs/20260201/confidence_comparison/` 下的 `REPORT.md`、`RESULTS.csv`、`SUMMARY.json`，包括逐源、整体、高低 rho 分组、oracle 差距和实测训练/预测耗时。跨模型家族的比较不属于单因素对照。
