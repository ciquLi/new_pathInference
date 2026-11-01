# pathInference 21／5：正式包名

正式方法目录为 `src/pathInference/`，从 `src/pathveil/` 实际移动而来。Python导入、子进程模块名、源码定位、安装元数据和当前文档统一使用 `pathInference`；正式入口仍为 `bash scripts/run_experiment.sh YYYYMMDD`。

本次沿用 `pats-v1.0-21x5-layout1` 的目录分类，将127个包内文件统一移入新目录；数值源码只做命名替换。21／5参数、日期配置、数据、模型、五日期结果及历史运行身份保持原样。原发布清单、原Git标签和数值来源清单保留历史路径，不随新名称重写。

`PATHVEIL_*`环境变量及 `_pathveil_metadata` 等旧模型字段保留，以兼容现有启动参数和已保存模型。GMPI的历史顶层模型模块名继续由 `core.activate()` 提供。

包名和源码路径参与代码身份，因此本次代码使用新的身份。旧目录的严格续跑检查不放宽；需要复核旧版本时使用原Git标签，不改写旧缓存的代码哈希。

本布局的[代码清单](pathInference-v1.0-21x5.json)单独登记当前文件哈希，并引用原始五日期基线清单。这里的版本名称是布局记录标识；本次没有创建或移动Git标签。

```bash
python -B tools/validation/verify_frozen_release.py
PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B -m pathInference --help
```

测试和March回归的记录见[验证摘要](pathInference-v1.0-21x5-validation.json)。回归仅重新计算冻结特征的模型分数和筛选结果；没有启动新日期实验或重训模型。
