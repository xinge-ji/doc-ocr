## 背景
- 任务：切换 PaddleOCR 3.3 API，修复 draw_ocr 导入报错，调整发票 pipeline 适配，并移除 paddle_vl.py。
- 相关文件：app/services/ocr/paddle_ocr.py，app/services/pipelines/invoice.py，app/services/ocr/paddle_vl.py，app/main.py（引用）。

## 计划
1) 调整 invoice pipeline 序列化逻辑，去除 block/line 排序依赖，保持按页顺序输出 bbox+text。
2) 删除 paddle_vl.py 文件并清理引用。
3) 自查全局依赖，确保 main 启动路径仅使用 PaddleOcrClient。

## 注意
- 环境读写受限，保存可视化使用 PaddleOCR predict 结果的 img/save_to_img。
