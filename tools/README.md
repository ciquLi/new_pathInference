# 独立维护工具

这些工具不属于日常PATS方法入口；正式运行脚本在 `scripts/`，方法实现位于 `src/pathveil/`，自动化测试位于 `tests/`。

| 目录／文件 | 用途 | 写入范围 |
|---|---|---|
| `validation/verify_frozen_release.py` | 核对当前布局版本的代码及可选基线资产哈希 | 只读取 |
| `validation/verify_migration.py` | 重放March冻结特征和评分，核验数值一致性 | 指定回归输出目录 |
| `data/check_data.py` | 检查指定日期输入是否就绪 | 仅指定 `--output` 时写报告 |
| `data/download_data.py` | 下载已登记的ASRank公开数据 | 项目 `data/` |
| `legacy/watch_adaptive_experiments.py` | 旧ρ流程的进度／结果对照维护 | 所指定旧运行的进度文档 |

旧watcher保留为历史工具，不能用于当前PATS进度报告。当前PATS进度由正式工作流写入。

```bash
# 仅读取代码和原发布清单
python -B tools/validation/verify_frozen_release.py

# 同时核验原发布登记的本地模型、候选和结果
python -B tools/validation/verify_frozen_release.py --assets

# 显式回归输出；不重训模型
/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B tools/validation/verify_migration.py --run-dir runs/verification/pats-v1.0-21x5-layout1/march_migration
```

默认核对目录分类后的 `pats-v1.0-21x5-layout1` 清单。原 `pats-v1.0-21x5` 清单按原样保留，反映原路径；核验原快照应在对应Git标签下执行原命令，不能将路径迁移误当作数值方法变化。
