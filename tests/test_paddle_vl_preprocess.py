import pytest

from app.services.ocr.paddle_vl import PaddleVLOcrClient


class DummyPreprocessor:
    def __init__(self) -> None:
        self.called_with: str | None = None

    def process(self, path: str):
        self.called_with = path
        return ["processed-path"], lambda: None


class DummyPipeline:
    def __init__(self) -> None:
        self.received: str | list[str] | None = None

    def predict(self, path: str):
        self.received = path
        return []


@pytest.mark.asyncio
async def test_extract_runs_preprocess_and_uses_processed_path(tmp_path):
    pre = DummyPreprocessor()
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
    assert pipeline.received == "processed-path"
    assert result.items == []
