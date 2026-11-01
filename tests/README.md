# 自动化测试

本目录只放软件测试。`test_*.py`中的合成路径、小数据和模拟模型用于检查行为，不是研究实验结果，也不会被正式方法导入。

```bash
PYTHONPATH=src /home/lxl/anaconda3/envs/rl-aspath-inference/bin/python -B -m unittest discover -s tests
```

| 文件组 | 检查对象 |
|---|---|
| `test_pats_*`、`test_full_tree_protocol.py` | 当前PATS策略、入库和完整树协议 |
| `test_neighbor_*`、`test_upstream_*`、`test_source_split*` | 邻居模型、上游身份和源隔离 |
| `test_downstream_*`、`test_heldout_*` | 下游训练输入与最终评测边界 |
| `test_*scheduler.py`、`test_runtime_resources.py` | 进程调度和资源管理 |
| `test_data_*`、`test_bgp_holdout.py` | 日期、数据准备和留出边界 |
| `test_traceroute_*` | 独立traceroute流程 |
| `test_frozen_release.py` | 稳定版本、文件哈希和历史清单不可替换 |
| `test_rho_*`、`test_core_paths.py` | 历史数值与兼容性 |

独立研究实验在 `research/`登记，准确率产物在 `runs/`；完整导航见 [DIRECTORY_MAP.md](../DIRECTORY_MAP.md)。
