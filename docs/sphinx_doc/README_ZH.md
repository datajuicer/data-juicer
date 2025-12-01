# Data-Juicer 文档

Data-Juicer 借助 Sphinx 构建文档。
如需更新生成的文档，请运行以下命令：

```bash
# $~/data_juicer/docs/sphinx_doc
# 1.安装 sphinx 的依赖并初始化 sphinx-quickstart
# 注意：请在python>=3.11环境下运行
uv pip install "py-data-juicer[dev]"
# 2. 运行文档构建脚本
./build_doc.sh

# 3. 构建完成的文档存储目录为 `build/`
mv build/ position_to_publish
```

Github上的自动化部署配置可参考 [该处](
https://github.com/datajuicer/data-juicer/blob/main/.github/workflows/deploy_sphinx_docs.yml).