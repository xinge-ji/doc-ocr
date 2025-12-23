任务：规则抽取增加区域过滤 + 位置按行匹配

目标：
- 文本锚点匹配仅在指定区域内执行，避免左右区域串行
- 位置抽取支持按行匹配（pos_match_scope=line）
- 更新模板 buyer/seller 区域与规则
- 补最小用例覆盖左右同一行不串

步骤：
1) invoice_rule_extractor.py: 解析区域边界、区域过滤文本锚点、pos_match_scope=line
2) vat_special_einvoice.json: buyer/seller 使用 pos + line，左右区域不重叠
3) tests: 增加或更新 buyer/seller 分离用例
