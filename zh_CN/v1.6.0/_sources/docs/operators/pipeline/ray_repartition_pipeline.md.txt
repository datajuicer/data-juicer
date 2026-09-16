# ray_repartition_pipeline

Repartition a Ray Dataset into a target number of blocks.

将 Ray Dataset 重新分区到指定 block 数。

This is a Ray-only dataset-level operator. It changes the number of Ray Dataset
blocks and fails fast when used with the local executor because local datasets
do not expose Ray Dataset blocks.

这是一个仅适用于 Ray 的数据集级算子。它会调整 Ray Dataset 的 block 数；如果在本地执行器中使用，会直接报错，因为本地数据集没有 Ray Dataset blocks。

Type 算子类型: **pipeline**

Tags 标签: ray, cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `num_blocks` | <class 'int'> | `1` | Target number of Ray Dataset blocks. |
| `shuffle` | <class 'bool'> | `False` | Whether to shuffle records during repartition. |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/pipeline/ray_repartition_pipeline.py)
- [unit test 单元测试](../../../tests/ops/pipeline/test_ray_repartition_pipeline.py)
- [Return operator list 返回算子列表](../../Operators.md)
