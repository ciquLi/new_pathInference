# PATS 21／5：目录分类版本

标签：`pats-v1.0-21x5-layout1`。基于 `pats-v1.0-21x5`；原标签、五日期基线结果及其发布清单保持原样。

本次只按用途分类：正式方法在 `src/pathveil/`，正式入口及运行支撑在 `scripts/`，自动化测试在 `tests/`，独立工具在 `tools/`，研究实验登记在 `research/`。完整导航见根目录 [DIRECTORY_MAP.md](../../DIRECTORY_MAP.md)。

迁移了5个独立脚本；数据检查／下载进入 `tools/data/`，版本核验／March回归进入 `tools/validation/`，旧ρ进度维护进入 `tools/legacy/`。需要相对路径定位项目根的脚本同步调整了一层目录；正式入口和调度脚本保留原路径。

`src/`、`configs/`和正式运行脚本与原稳定标签一致。新增研究区只有说明，未启动新研究实验。原模型、候选、结果、运行指针和历史身份未改写。

[新布局机器清单](pats-v1.0-21x5-layout1.json)登记195个文件，并通过原发布清单的SHA256引用原资产；原清单不重写。新的只读核验工具会检查这一父清单，测试也覆盖父清单缺失、被修改和资产损坏的情形。

```bash
python -B tools/validation/verify_frozen_release.py
python -B tools/validation/verify_frozen_release.py --assets
```

新命令适用于当前布局；原发布说明中的旧命令属于原Git标签。

验收：236项测试，1项按原设计跳过；March全10源模型评分和发布CSV一致；所有迁移的CLI从项目外目录启动也能正确定位项目根。详细证据见[验证摘要](pats-v1.0-21x5-layout1-validation.json)。此次没有重新读取原3410个资产的全部内容做哈希，沿用原发布已通过的核验记录；源代码和目录迁移单独核验。
