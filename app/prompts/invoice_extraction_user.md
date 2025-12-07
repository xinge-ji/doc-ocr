基于下方 OCR 结果提取发票字段并返回严格的 JSON。

必须返回单个 JSON 对象，键名和层级固定：
{{
  "invoice_type": "...",
  "buyer": {{
    "name": null,
    "tax_id": null,
    "address": null,
    "phone": null,
    "bank_account_name": null,
    "bank_account_no": null
  }},
  "seller": {{
    "name": "",
    "tax_id": null,
    "address": null,
    "phone": null,
    "bank_account_name": null,
    "bank_account_no": null
  }},
  "issue_date": "YYYY-MM-DD",
  "issuer": "",
  "invoice_number": "",
  "lines": [
    {{
      "name": "",
      "spec": null,
      "unit": null,
      "quantity": null,
      "unit_price": null,
      "amount": 0,
      "tax_rate": null,
      "tax_amount": null,
      "line_total_with_tax": null
    }}
  ],
  "total_amount": null,
  "amount_with_tax": 0
}}

规则：
- 所有键必须出现；缺失信息用 null（可选字段）或空字符串（必填但未知）。
- 货币/数值用数字，日期格式 YYYY-MM-DD。
- **数量、单价等字段请去除单位（如“台”、“个”），只保留纯数字。**
- 输出必须是有效 JSON，不要有额外文本。

OCR 结果 (text + bbox[x1,y1,x2,y2] + page + ids)：
{ocr_json}
