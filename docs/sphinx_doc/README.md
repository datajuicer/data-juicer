# Data-Juicer Documentation

We build our documentation with help of Sphinx.
To update the generated
doc, please run the following commands:

```bash
# $~/data_juicer/docs/sphinx_doc
# 1. install the sphinx requirements and init the sphinx-quickstart
# Note: Please run in Python>=3.11 environment
uv pip install "py-data-juicer[dev]"

# 2. auto generate and build the doc
./build_doc.sh

# 3. finalize the doc, which is stored in the `build/` directory
mv build/ position_to_publish
```

Automatic action in github can be found in [here](https://github.com/datajuicer/data-juicer/blob/main/.github/workflows/deploy_sphinx_docs.yml).