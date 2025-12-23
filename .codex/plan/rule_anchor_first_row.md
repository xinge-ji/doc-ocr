# 任务：表格首行仅必填字段视为 anchor

## 上下文
- 表头后第一条数据行可能只有 name
- 需要把该行当作 anchor 并合并后续行
- 适用于所有 anchor 模式模板

## 计划
1. invoice_rule_extractor: 首行必填字段命中时放宽 anchor 判定
2. tests: 覆盖首行仅 name 的合并场景
3. README: 记录表格抽取行为
