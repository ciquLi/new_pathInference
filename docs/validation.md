> 本文为2026-10-24历史整理记录。2026-10-27日期化ρ流程的验收与边界见`runs/verification/rho_pipeline_20261027/REPORT.md`，当前运行方法见[rho_workflow.md](rho_workflow.md)。

# 2026-10-24 整理验收

旧工程已迁至同级 `MetaGMPI_legacy_20261024`，包含原 Git 历史、旧流程、中间试验和原结果。
新项目不依赖该目录。当前论文保持原内容和相对路径。

已完成的检查：

- 整理前冻结 2,161 份源码，核验 5,107 个模型或输入文件哈希。
- 正式方法的 366 个必需模型和工件迁入规范目录，源/目标哈希相同。
- 0301 全 10 个 AS 重新计算路径分数、pure 选择和统一路径库；分数逐位一致，CSV 字节一致。
- 50 行候选重新构建 222 维基础特征和 33 维目的地特征，逐位一致；root 和 frontier 计划一致。
- 在旧工程移出后，由正式 shell 入口完成全十源缓存特征回归。
- 新 core 可以加载原 GMPI v2 checkpoint，词表哈希、74,874 个 token、图节点和边计数一致。
- 独立重评分 685,765 个固定 AS pair，原基线 Jaccard 为 0.8259929786442877。
- traceroute 独立 scorer 在冻结预测上的 22 个共有指标与原 evaluator 完全一致。
- 0501、0601 官方 ASRank nodes/links 已完整下载，所有分页日期、总数、唯一性和 SHA256 通过。

这些是**整理与迁移验证**，不是新月份的实验结果。新日期的完整模型训练、PG 和 GMPI
尚未完成端到端运行；普通日期入口已实现完整阶段，正式结果以独立审计完成标记为准。

持久证据在 `.reproducibility/`，详细回归产物在 `runs/`。单元测试命令：

```bash
PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -m unittest discover -s tests -v
```
