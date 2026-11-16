# 两AS部分吸纳原始TRAIN路径

分支 `research/partial-train-v1`，正式代码恢复点 `9ce7a82`。只针对 AS34927、AS58057，日期为0201、0301、0401、0501、0601。数值基线使用原21／5固定运行，正式流程文件不改动。

原始TRAIN指其他原始训练源路径中、从目标AS开始的正向连续子路径；目标AS自身的原始观测仍保持留出。已有独立来源审计证明这些路径与对应TRAIN源CSV、展开集合一致，筛选只读该审计及经哈希核验的CSV。

首轮规则在新评测前固定：[CONFIG.json](CONFIG.json)。全部保留基线增强库，只吸纳匹配原前缀树、无环、长度2至20且无已知商业关系冲突的TRAIN路径；unknown保留。新增上限为原增强数的25%（向下取整），每目的地最多新增一条。优先补充增强库未覆盖的目的地，各首跳轮流选；同目的地先选最短路径，再按固定哈希打破并列。不使用测试路径、测试分布或测试准确率选路径。

25%是首轮保守的实验比例，不是已验证的最优比例。原基线测试分布已在此前诊断中查看，因此这次属于同数据集的探索性比较，不作为未经接触数据的最终验证。

训练标签与PATS本地知识均使用同一混合库哈希，使用原完整前缀树及原候选生成函数；仅更新该树的库绑定，几何结构不变。GMPI冷启动训练，保留原配置及pair-disjoint内部验证，选定模型后再读取最终路径计算指标。输入库发生变化，需要生成对应的新候选；原模型、树、增强库和大文件复用经核验的数据，不重跑上游。

薄入口：

```bash
bash research/partial_train_v1/run.sh
```

数值实现位于 `src/pathInference/research/partial_train.py`，正式入口不会导入它。输出为 `runs/YYYYMMDD/partial_train_v1/fixed25/`，总进度和对照表为 `runs/batches/partial_train_v1/`。日期按顺序执行；每日期只生成两个AS的候选。GMPI使用GPU，候选生成使用CPU。可用相同入口续跑，已完成阶段均需校验身份和输出哈希。

五日期首轮现已全部完成。宏平均Jaccard：AS34927为83.4508% → 85.1859%（+1.7351个百分点），AS58057为76.1315% → 72.8215%（-3.3100个百分点）。AS34927前3期提高、后2期略降，AS58057五期均下降；暂不将此规则并入正式方法。

最终审计及五日期逐查询诊断使用独立维护工具，不启动训练、不改写输入和历史基线：

```bash
python -B tools/validation/verify_partial_train.py
```

最终报告为 `runs/batches/partial_train_v1/REPORT.md`；`FINAL_AUDIT.json`记录参考/查询身份、独立重评分、树、库、检查点边界；`FINAL_SUMMARY.json`记录完整数值及宏平均。`PAIRED_DIAGNOSTICS.json`和`NEW_DESTINATION_DIAGNOSTICS.json`提供后验解释，不能用于自动调整已固定的吸纳规则。
