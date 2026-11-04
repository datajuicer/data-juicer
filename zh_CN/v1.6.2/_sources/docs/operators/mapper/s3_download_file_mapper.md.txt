# s3_download_file_mapper

Mapper to download files from S3 to local files or load them into memory.

This operator downloads files from S3 URLs (s3://...) or handles local files. It supports:
- Downloading multiple files concurrently
- Saving files to a specified directory or loading content into memory
- Resume download functionality
- S3 authentication with access keys
- Custom S3 endpoints (for S3-compatible services like MinIO)

The operator processes nested lists of URLs/paths, maintaining the original structure in the output.

用于从 S3 下载文件到本地或加载到内存的 Mapper。

该算子可从 S3 URL（s3://...）下载文件，也支持处理本地文件。功能包括：  
- 并发下载多个文件  
- 将文件保存到指定目录或将内容加载到内存  
- 支持断点续传  
- 使用访问密钥进行 S3 身份验证  
- 支持自定义 S3 端点（适用于 MinIO 等 S3 兼容服务）  

该算子可处理嵌套的 URL/路径列表，并在输出中保持原始结构。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `download_field` | <class 'str'> | `None` | The field name to get the URL/path to download. |
| `save_dir` | <class 'str'> | `None` | The directory to save downloaded files. |
| `save_field` | <class 'str'> | `None` | The field name to save the downloaded file content. |
| `resume_download` | <class 'bool'> | `False` | Whether to resume download. If True, skip the sample if it exists. |
| `timeout` | <class 'int'> | `30` | (Deprecated) Kept for backward compatibility, not used for S3 downloads. |
| `max_concurrent` | <class 'int'> | `10` | Maximum concurrent downloads. |
| `aws_access_key_id` | <class 'str'> | `None` | AWS access key ID for S3. |
| `aws_secret_access_key` | <class 'str'> | `None` | AWS secret access key for S3. |
| `aws_session_token` | <class 'str'> | `None` | AWS session token for S3 (optional). |
| `aws_region` | <class 'str'> | `None` | AWS region for S3. |
| `endpoint_url` | <class 'str'> | `None` | Custom S3 endpoint URL (for S3-compatible services). |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/s3_download_file_mapper.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)