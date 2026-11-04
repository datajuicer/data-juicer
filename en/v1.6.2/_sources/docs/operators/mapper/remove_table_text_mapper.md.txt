# remove_table_text_mapper

Mapper to remove table texts from text samples.

This operator uses regular expressions to identify and remove tables from the text. It targets tables with a specified range of columns, defined by the minimum and maximum number of columns. The operator iterates over each sample, applying the regex pattern to remove tables that match the column criteria. The processed text, with tables removed, is then stored back in the sample. This operation is batched for efficiency.

用于从文本样本中移除表格文本的映射器。

该算子使用正则表达式来识别并移除文本中的表格。它针对具有指定列数范围的表格，该范围由最小和最大列数定义。算子遍历每个样本，应用正则表达式模式以移除符合列标准的表格。处理后的文本（已移除表格）将存回样本中。此操作为了效率进行了批处理。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_col` | typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=2), Le(le=20)])] | `2` | The min number of columns of table to remove. |
| `max_col` | typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=2), Le(le=20)])] | `20` | The max number of columns of table to remove. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_single_table_case
```python
RemoveTableTextMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a table:
编号 分行 营运资金1 营运资金2 营运资金3 营运资金4 营运资金5
① 北京分行 495,000,000.00 200,000,000.00 295,000,000.00 - 495,000,000.00
② 大连分行 440,000,000.00 100,000,000.00 340,000,000.00 - 440,000,000.00
③ 重庆分行 500,000,000.00 100,000,000.00 400,000,000.00 - 500,000,000.00
④ 南京分行 430,000,000.00 100,000,000.00 330...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (119 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a table:
编号 分行 营运资金1 营运资金2 营运资金3 营运资金4 营运资金5
① 北京分行 495,000,000.00 200,000,000.00 295,000,000.00 - 495,000,000.00
② 大连分行 440,000,000.00 100,000,000.00 340,000,000.00 - 440,000,000.00
③ 重庆分行 500,000,000.00 100,000,000.00 400,000,000.00 - 500,000,000.00
④ 南京分行 430,000,000.00 100,000,000.00 330,000,000.00 - 430,000,000.00
⑤ 青岛分行 500,000,000.00 - 100,159,277.60 399,840,722.40 500,000,000.00
The end of the table.</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a table:
The end of the table.</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to remove a typical table from the text. The input text contains a table with multiple rows and columns, which is completely removed by the operator, leaving only the introductory and concluding sentences. This showcases the operator's effectiveness in identifying and removing structured data (tables) from unstructured text.
这个例子展示了算子从文本中移除典型表格的能力。输入文本包含一个有多行多列的表格，算子完全移除了这个表格，只留下了引言和结尾句。这展示了算子在识别并移除结构化数据（表格）方面是有效的。

### test_false_positive_case
```python
RemoveTableTextMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">平安银行股份有限公司
非公开发行普通股认购资金到位情况的验资报告非公开发行普通股认购资金到位情况的验资报告
普华永道中天验字(2015)第446号
(第一页，共二页)
平安银行股份有限公司：
平安银行股份有限公司(以下简称“贵公司”)委托中信证券股份有限公司作为主
承销商非公开发行普通股 598,802,395 股。我们接受委托，审验了贵公司截至
2015年 5月 5日止由中信证券股份有限公司代收取的向境内合格投资者非公开发行
普通股认购资金的到位情况。按照国家相关法律、法规的规定以及认购协议、合同
的要求出资认购，提供真实、合法、完整的验资资料，保护资产的安全、完整是贵
公司管理层及中信证券...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1997 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">平安银行股份有限公司
非公开发行普通股认购资金到位情况的验资报告非公开发行普通股认购资金到位情况的验资报告
普华永道中天验字(2015)第446号
(第一页，共二页)
平安银行股份有限公司：
平安银行股份有限公司(以下简称“贵公司”)委托中信证券股份有限公司作为主
承销商非公开发行普通股 598,802,395 股。我们接受委托，审验了贵公司截至
2015年 5月 5日止由中信证券股份有限公司代收取的向境内合格投资者非公开发行
普通股认购资金的到位情况。按照国家相关法律、法规的规定以及认购协议、合同
的要求出资认购，提供真实、合法、完整的验资资料，保护资产的安全、完整是贵
公司管理层及中信证券股份有限公司的责任。我们的责任是对贵公司由中信证券股
份有限公司代收取的境内合格投资者本次非公开发行普通股认购资金的到位情况发
表审验意见。我们的审验是依据《中国注册会计师审计准则第 1602号——验资》进
行的。在审验过程中，我们结合贵公司的实际情况，实施了检查等必要的审验程
序。
经贵公司 2014 年 7 月 15 日第九届董事会第五次会议提议，2014 年 8 月 4 日
2014 年第二次临时股东大会审议通过《平安银行股份有限公司关于非公开发行普
通股方案的议案》，贵公司拟向境内合格投资者非公开发行不超过 1,070,663,811
股普通股。根据中国证券监督管理委员会证监许可[2015]697 号文《关于核准平安
银行股份有限公司非公开发行股票的批复》，贵公司获准向境内合格投资者非公开
发行不超过1,070,663,811股普通股。普华永道中天验字(2015)第446号
(第二页，共二页)
经我们审验，截至 2015 年 5 月 5 日止，贵公司以每股人民币 16.70 元合计向
境内合格投资者非公开发行普通股 598,802,395 股，由发行主承销商中信证券股份
有限公司代贵公司实际收到人民币 9,999,999,996.50元。所有认购资金均以人民币
现金形式汇入。
本验资报告仅供贵公司向中国证券监督管理委员会、深圳证券交易所报送资料
及向中国证券登记结算有限责任公司深圳分公司申请非公开发行普通股登记时使
用，不应将其视为是对贵公司验资报告日后资本保全、偿债能力和持续经营能力等
的保证。因使用本验资报告不当造成的后果，与执行本验资业务的注册会计师及会
计师事务所无关。
附件一 非公开发行普通股认购资金到位情况明细表
附件二 验资事项说明
附件三 普华永道中天会计师事务所(特殊普通合伙)营业执照
附件四 普华永道中天会计师事务所(特殊普通合伙)执业证书
附件五 普华永道中天会计师事务所(特殊普通合伙)证券相关业务许可证
普华永道中天会计师事务所 注册会计师
(特殊普通合伙) 姚文平
中国•上海市 注册会计师
2015年 5月7日 朱丽平
2附件一
非公开发行普通股认购资金到位情况明细表
截至2015年5月 5日止
被审验单位名称：平安银行股份有限公司
货币单位：人民币元
金额
到位认购资金 9,999,999,996.50
3附件二
验资事项说明
一、 基本情况
平安银行股份有限公司(以下简称“贵公司”)是中国平安保险(集团)股份有限公司控股的
一家跨区域经营的股份制商业银行，是原深圳发展银行股份有限公司以吸收合并原平
安银行股份有限公司的方式完成两行整合并更名的银行，总部位于深圳。原深圳发展
银行成立于 1987年 12月 22日，并于 1991年 4月 3日在深圳证券交易所上市(股票代
码：000001)。
贵公司经中国银行业监督管理委员会批准领有 00386413 号金融许可证，机构编码为
B0014H144030001，深圳市工商行政管理局批准核发的 440301103098545 号《中华
人民共和国企业法人执照》。贵公司注册资本为人民币 13,709,873,744 元，实收资本
(股本)为人民币 13,709,873,744元，其中包括有限售条件股份 1,905,819,165 股，无限
售条件股份 11,804,054,579 股。贵公司的上述实收资本(股本)已经普华永道中天会计
师事务所(特殊普通合伙)审验，并已于 2015 年 4 月 13 日出具普华永道中天验字(2015)
第321号验资报告。
二、本次非公开发行普通股审批及情况说明
于 2014 年 7 月 15 日第九届董事会第五次会议《平安银行股份有限公司关于非公开发
行普通股方案的议案》，同意提议股东大会批准贵公司向境内合格投资者非公开发行
不超过 1,070,663,811 股普通股。于 2014 年 8 月 4 日 2014 年第二次临时股东大会审
议通过，批准了董事会的上述提议。中国证券监督管理委员会于 2015 年 4 月 22 日出
具证监许可[2015]697 号文《关于核准平安银行股份有限公司非公开发行股票的批
复》核准了贵公司向境内合格投资者非公开发行不超过1,070,663,811股普通股。
三、 审验结果
经我们审验，截至 2015 年 5 月 5 日止，贵公司已完成普通股 598,802,395 股的发
行，每股发行价格为人民币 16.70 元，认购资金合计人民币 9,999,999,996.50 元，全
部以人民币现金形式汇入，由发行主承销商中信证券股份有限公司代贵公司收缴，已
全部存入主承销商中信证券股份有限公司于平安银行股份有限公司北京分行营业部开
立的19014508950004银行账号内。
4</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">平安银行股份有限公司
非公开发行普通股认购资金到位情况的验资报告非公开发行普通股认购资金到位情况的验资报告
普华永道中天验字(2015)第446号
(第一页，共二页)
平安银行股份有限公司：
平安银行股份有限公司(以下简称“贵公司”)委托中信证券股份有限公司作为主
普通股认购资金的到位情况。按照国家相关法律、法规的规定以及认购协议、合同
的要求出资认购，提供真实、合法、完整的验资资料，保护资产的安全、完整是贵
公司管理层及中信证券股份有限公司的责任。我们的责任是对贵公司由中信证券股
份有限公司代收取的境内合格投资者本次非公开发行普通股认购资金的到位情况发
表审验意见。我们的审验是依据《中国注册...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1474 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">平安银行股份有限公司
非公开发行普通股认购资金到位情况的验资报告非公开发行普通股认购资金到位情况的验资报告
普华永道中天验字(2015)第446号
(第一页，共二页)
平安银行股份有限公司：
平安银行股份有限公司(以下简称“贵公司”)委托中信证券股份有限公司作为主
普通股认购资金的到位情况。按照国家相关法律、法规的规定以及认购协议、合同
的要求出资认购，提供真实、合法、完整的验资资料，保护资产的安全、完整是贵
公司管理层及中信证券股份有限公司的责任。我们的责任是对贵公司由中信证券股
份有限公司代收取的境内合格投资者本次非公开发行普通股认购资金的到位情况发
表审验意见。我们的审验是依据《中国注册会计师审计准则第 1602号——验资》进
行的。在审验过程中，我们结合贵公司的实际情况，实施了检查等必要的审验程
序。
经贵公司 2014 年 7 月 15 日第九届董事会第五次会议提议，2014 年 8 月 4 日
银行股份有限公司非公开发行股票的批复》，贵公司获准向境内合格投资者非公开
发行不超过1,070,663,811股普通股。普华永道中天验字(2015)第446号
(第二页，共二页)
经我们审验，截至 2015 年 5 月 5 日止，贵公司以每股人民币 16.70 元合计向
境内合格投资者非公开发行普通股 598,802,395 股，由发行主承销商中信证券股份
有限公司代贵公司实际收到人民币 9,999,999,996.50元。所有认购资金均以人民币
现金形式汇入。
本验资报告仅供贵公司向中国证券监督管理委员会、深圳证券交易所报送资料
及向中国证券登记结算有限责任公司深圳分公司申请非公开发行普通股登记时使
用，不应将其视为是对贵公司验资报告日后资本保全、偿债能力和持续经营能力等
的保证。因使用本验资报告不当造成的后果，与执行本验资业务的注册会计师及会
计师事务所无关。
2015年 5月7日 朱丽平
2附件一
非公开发行普通股认购资金到位情况明细表
截至2015年5月 5日止
被审验单位名称：平安银行股份有限公司
货币单位：人民币元
金额
到位认购资金 9,999,999,996.50
3附件二
验资事项说明
一、 基本情况
平安银行股份有限公司(以下简称“贵公司”)是中国平安保险(集团)股份有限公司控股的
一家跨区域经营的股份制商业银行，是原深圳发展银行股份有限公司以吸收合并原平
安银行股份有限公司的方式完成两行整合并更名的银行，总部位于深圳。原深圳发展
银行成立于 1987年 12月 22日，并于 1991年 4月 3日在深圳证券交易所上市(股票代
码：000001)。
(股本)为人民币 13,709,873,744元，其中包括有限售条件股份 1,905,819,165 股，无限
售条件股份 11,804,054,579 股。贵公司的上述实收资本(股本)已经普华永道中天会计
师事务所(特殊普通合伙)审验，并已于 2015 年 4 月 13 日出具普华永道中天验字(2015)
第321号验资报告。
二、本次非公开发行普通股审批及情况说明
于 2014 年 7 月 15 日第九届董事会第五次会议《平安银行股份有限公司关于非公开发
行普通股方案的议案》，同意提议股东大会批准贵公司向境内合格投资者非公开发行
不超过 1,070,663,811 股普通股。于 2014 年 8 月 4 日 2014 年第二次临时股东大会审
议通过，批准了董事会的上述提议。中国证券监督管理委员会于 2015 年 4 月 22 日出
具证监许可[2015]697 号文《关于核准平安银行股份有限公司非公开发行股票的批
复》核准了贵公司向境内合格投资者非公开发行不超过1,070,663,811股普通股。
三、 审验结果
经我们审验，截至 2015 年 5 月 5 日止，贵公司已完成普通股 598,802,395 股的发
行，每股发行价格为人民币 16.70 元，认购资金合计人民币 9,999,999,996.50 元，全
部以人民币现金形式汇入，由发行主承销商中信证券股份有限公司代贵公司收缴，已
全部存入主承销商中信证券股份有限公司于平安银行股份有限公司北京分行营业部开
立的19014508950004银行账号内。
4</pre></details></div>

#### ✨ explanation 解释
This example illustrates an important edge case where the input text does not contain a table but has a format that could be mistaken for one. The operator correctly identifies that there is no table present and leaves the text unchanged. This shows the operator's capability to distinguish between actual tables and text that may resemble a table, preventing false positives.
这个例子展示了一个重要的边界情况，其中输入文本不包含表格，但格式可能被误认为是表格。算子正确地识别出没有表格存在，并且保持文本不变。这展示了算子能够区分实际的表格和看起来像表格的文本，防止了误报。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_table_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_table_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)