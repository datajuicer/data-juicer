# s3_upload_file_mapper

Mapper to upload local files to S3 and update paths to S3 URLs.

This operator uploads files from local paths to S3 storage. It supports:
- Uploading multiple files concurrently
- Updating file paths in the dataset to S3 URLs
- Optional deletion of local files after successful upload
- Custom S3 endpoints (for S3-compatible services like MinIO)
- Skipping already uploaded files (based on S3 key)

The operator processes nested lists of paths, maintaining the original structure in the output.

用于将本地文件上传至 S3 并将路径更新为 S3 URL 的 Mapper。

该算子将本地路径的文件上传至 S3 存储，支持以下功能：  
- 并发上传多个文件  
- 更新数据集中的文件路径为 S3 URL  
- 可选在成功上传后删除本地文件  
- 支持自定义 S3 端点（适用于 MinIO 等 S3 兼容服务）  
- 跳过已上传的文件（基于 S3 key 判断）  

该算子可处理嵌套的路径列表，并在输出中保持原始结构。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `upload_field` | <class 'str'> | `None` | The field name containing file paths to upload. |
| `s3_bucket` | <class 'str'> | `None` | S3 bucket name to upload files to. |
| `s3_prefix` | <class 'str'> | `''` | Prefix (folder path) in S3 bucket. E.g., 'videos/' or 'data/videos/'. |
| `aws_access_key_id` | <class 'str'> | `None` | AWS access key ID for S3. |
| `aws_secret_access_key` | <class 'str'> | `None` | AWS secret access key for S3. |
| `aws_session_token` | <class 'str'> | `None` | AWS session token for S3 (optional). |
| `aws_region` | <class 'str'> | `None` | AWS region for S3. |
| `endpoint_url` | <class 'str'> | `None` | Custom S3 endpoint URL (for S3-compatible services). |
| `remove_local` | <class 'bool'> | `False` | Whether to delete local files after successful upload. |
| `skip_existing` | <class 'bool'> | `True` | Whether to skip uploading if file already exists in S3. |
| `max_concurrent` | <class 'int'> | `10` | Maximum concurrent uploads. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/s3_upload_file_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_s3_upload_file_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)