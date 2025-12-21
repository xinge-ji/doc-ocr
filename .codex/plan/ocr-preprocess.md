# OCR 前处理增强计划

## 背景
- 任务：为发票 OCR pipeline 添加前处理（自动旋转、去扭曲、通用增强），不改 PP-DocLayout。
- 目标：在 PaddleVLOcrClient 调用前串联 DocPreprocessor + 自研增强，增加配置开关。

## 方案
- 方案A：DocPreprocessor(use_doc_orientation_classify/use_doc_unwarping) + OpenCV 基础增强（降噪+CLAHE/阈值），由 OCR 客户端持有和调用。

## 任务拆解
1) 梳理配置与依赖（完成）。
2) 新建 preprocess 模块封装 DocPreprocessor + OpenCV 增强，支持 path/bytes，输出临时文件，异常回退源图。
3) 在 paddle_vl.py 里接入前处理，新增配置开关，管理清理。
4) 更新 config/.env.example/README，补测试（mock preprocessor 调用）。

## 风险/注意
- DocPreprocessor 启用需模型可用，性能影响需可配置。
- OpenCV 依赖体积较大，需确认可接受；增强开关默认开可按需关闭。
- 临时文件清理确保不泄漏。

## 待确认
- 依赖添加 opencv-python 是否 OK（默认采用）。

