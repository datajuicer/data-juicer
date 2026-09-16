# API服务化
中文 | [English Page](DJ_service.md) 

为了进一步提升Data-Juicer用户体验，我们新增了基于 API 的服务功能（Service API），使用户能够以更便捷的方式集成和使用 Data-Juicer 的强大算子池。通过该服务功能，用户无需深入了解框架的底层实现细节，即可快速构建数据处理流水线，并与现有系统无缝对接。用户也可通过该服务实现不同project之间的环境隔离。本文档将详细介绍如何启动和使用这一服务功能，帮助您快速上手并充分发挥 Data-Juicer 的潜力。

## 启动服务
执行如下代码：
```bash
uvicorn service:app
```

## API调用
API支持调用Data-Juicer所有`__init__.py`中的函数和类（调用类的某个函数）。函数调用GET调用，类通过POST调用。

### 协议

#### url路径
采用GET调用函数，url路径与Data-Juicer库中函数引用路径一致，如`from data_juicer.config import init_configs`对应路径为`data_juicer/config/init_configs`。采用POST调用类的某个函数，url路径与Data-Juicer库中类路径拼接上函数名，如调用`TextLengthFIlter`算子的`compute_stats_batched`函数，对应路径为`data_juicer/ops/filter/TextLengthFilter/compute_stats_batched`。

#### 参数
进行GET和POST调用时，会自动将参数转化为list，同时，查询参数不支持字典传输。于是，如果传递的参数的value是list或dict，我们统一用`json.dumps`传输，并在前面添加特殊符号`<json_dumps>`与一般的string进行区分。

#### 特例
1. 针对`cfg`参数，我们默认采用`json.dumps`传输，无需添加特殊符号`<json_dumps>`。
2. 针对`dataset`参数，允许用户传输`dataset`在服务器上的路径，sever将加载dataset。
3. 允许用户设定`skip_return`参数，为`True`时将不返回函数调用的结果，避免一些网络无法传输带来的错误。

### 函数调用
采用GET调用，url路径与Data-Juicer库中函数引用路径一致，查询参数用来传递函数的参数。

例如，可用如下python代码调用Data-Juicer的参数init函数`init_configs`，获取Data-Juicer所有参数。

```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/config/init_configs'
params = {"args": json_prefix + json.dumps(['--config', './configs/demo/process.yaml'])}
response = requests.get(url, params=params)
print(json.loads(response.text))
```

对应的curl代码如下：

```bash
curl -G "http://localhost:8000/data_juicer/config/init_configs" \
     --data-urlencode "args=--config" \
     --data-urlencode "args=./configs/demo/process.yaml"
```

### 类的函数调用
采用POST调用，url路径与Data-Juicer库中类路径拼接上函数名，查询参数用来传递函数的参数，JSON字段用来传递类构造函数所需参数。

例如，可用如下python代码调用Data-Juicer的`TextLengthFIlter`算子。
```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched'
params = {'samples': json_prefix + json.dumps({'text': ['12345', '123'], '__dj__stats__': [{}, {}]})}
init_json = {'min_len': 4, 'max_len': 10}
response = requests.post(url, params=params, json=init_json)
print(json.loads(response.text))
```

对应的curl代码如下：

```bash
curl -X POST \
  "http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched?samples=%3Cjson_dumps%3E%7B%22text%22%3A%20%5B%2212345%22%2C%20%22123%22%5D%2C%20%22__dj__stats__%22%3A%20%5B%7B%7D%2C%20%7B%7D%5D%7D" \
  -H "Content-Type: application/json" \
  -d '{"min_len": 4, "max_len": 10}'
```


注：如果需要调用`Executor`类或`Analyzer`类的`run`函数进行数据处理和数据分析，需要先调用`init_configs`或`get_init_configs`函数获取完整的Data-Juicer参数来构造这两个类。具体可参考如下演示。

## 演示
我们结合[AgentScope](https://github.com/modelscope/agentscope)实现了用户通过自然语言调用Data-Juicer算子进行数据清洗的功能，算子采用API服务的方式进行调用。具体代码请参考[这里](../demos/api_service)。
