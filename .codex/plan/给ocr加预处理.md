任务：给 OCR 模块增加预处理，使用 PaddleOCR 的 DocImgOrientationClassification 自动纠偏，旋转后的图片持久化供后续 OCR 调用，支持 PDF 分页转换。

上下文：
- 代码结构：OCR 客户端在 app/services/ocr/paddle_vl.py，BaseOcrClient 定义于 app/services/ocr/base.py；OCR schema 在 app/schemas/ocr.py；配置在 app/core/config.py。
- 依赖：pyproject 已包含 paddleocr[doc-parser]、opencv-python，可用 DocImgOrientationClassification 和 pdf 转图工具。
- 需求：自动方向检测并旋转，图片/PDF 支持；旋转后文件落盘（不清理），返回路径供 OCR；路径/命名按最佳实践。

方案（执行路线）：
1) 预处理模块新增：app/services/ocr/preprocess.py
   - 封装 DocImgOrientationClassification，懒加载模型。
   - 输入 source (path/bytes)、filename、content_type。
   - PDF：使用 paddleocr 提供的 pdf→image（若不可用则 fallback cv2+fitz，但优先 paddleocr）；得到每页图片路径。
   - 对每页图片跑 orientation predict，取 label_names -> angle（0/90/180/270）。
   - 用 cv2 旋转并保存到 debug/preprocessed/{YYYYMMDD}/{uuid}/{page}_rot{angle}.jpg；返回路径列表和角度元数据。
2) OCR 集成：调整 app/services/ocr/paddle_vl.py
   - 在 extract 中调用预处理获取旋转后图片路径列表，逐个喂 pipeline.predict。
   - 解析逻辑复用现有 _parse_single_page_dict。
   - 预处理输出目录可配置（新增 settings 字段，默认 debug/preprocessed）。
3) 配置：在 app/core/config.py 增加 preprocess_output_dir 默认值；确保不影响现有必填字段。
4) 测试：在 tests 目录添加/调整单测
   - Mock DocImgOrientationClassification.predict 返回 90/180 等标签。
   - Mock cv2.imwrite / rotation，验证路径命名和角度选择。
   - PDF 多页路径生成与返回列表长度。
5) 说明：必要时在 README 或注释标注预处理行为与输出位置。

后续请求：
- 需要写权限已申请；按上述步骤编码。
