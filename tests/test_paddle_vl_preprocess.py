import pytest

from app.services.ocr.paddle_vl import PaddleVLOcrClient


class DummyPreprocessor:
    def __init__(self, processed_path: str) -> None:
        self.called_with: str | None = None
        self.processed_path = processed_path

    def process(self, path: str):
        self.called_with = path
        return [self.processed_path], lambda: None


class DummyPipeline:
    def __init__(self) -> None:
        self.received: str | list[str] | None = None

    def predict(self, path: str):
        self.received = path
        return []


@pytest.mark.asyncio
async def test_extract_runs_preprocess_and_uses_processed_path(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed.write_bytes(b"processed")
    pre = DummyPreprocessor(str(processed))
    pipeline = DummyPipeline()

    client = PaddleVLOcrClient(
        model_dir=tmp_path,
        backend="dummy",
        server_url="http://dummy",
        layout_model_name="dummy",
        pipeline=pipeline,  # type: ignore[arg-type]
        preprocessor=pre,
    )

    result = await client.extract(b"binarydata", filename="demo.jpg", content_type="image/jpeg")

    assert pre.called_with is not None
    assert pipeline.received == str(processed)
    assert result.items == []


@pytest.mark.asyncio
async def test_debug_save_base64_creates_files(tmp_path):
    raw_file = tmp_path / "raw.jpg"
    raw_file.write_bytes(b"rawdata")
    pre = DummyPreprocessor(str(raw_file))
    pipeline = DummyPipeline()
    debug_dir = tmp_path / "debug"

    client = PaddleVLOcrClient(
        model_dir=tmp_path,
        backend="dummy",
        server_url="http://dummy",
        layout_model_name="dummy",
        pipeline=pipeline,  # type: ignore[arg-type]
        preprocessor=pre,
        debug_save_base64=True,
        debug_save_dir=debug_dir,
    )

    await client.extract(b"binarydata", filename="demo.jpg", content_type="image/jpeg")

    saved_files = list(debug_dir.glob("*/page_0.b64"))
    assert saved_files, "debug base64 file not created"
    content = saved_files[0].read_text()
    assert content, "debug base64 content empty"
