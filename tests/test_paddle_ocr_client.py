import pytest

from app.schemas.ocr import BoundingBox
from app.services.ocr.paddle_ocr import PaddleOcrClient


class FakeOcr:
    def __init__(self, result):
        self._result = result

    def ocr(self, _path, cls=False):
        return self._result


class FakePreprocessor:
    class _Page:
        def __init__(self, path, page):
            self.path = path
            self.page = page

    class _Result:
        def __init__(self, pages):
            self.pages = pages

    def __init__(self, path: str):
        self.path = path

    def preprocess(self, *_args, **_kwargs):
        return self._Result([self._Page(self.path, 1)])


@pytest.mark.asyncio
async def test_paddle_ocr_client_parses_bbox_and_text(monkeypatch):
    fake_result = [
        [
            (
                [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                ("hello", 0.99),
            ),
            (
                [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]],
                ("world", 0.88),
            ),
        ]
    ]
    client = PaddleOcrClient(
        ocr=FakeOcr(fake_result),
        preprocessor=FakePreprocessor("dummy.jpg"),
        save_visualization=False,
    )

    result = await client.extract(b"dummy-bytes")

    assert len(result.items) == 2
    first = result.items[0]
    assert first.text == "hello"
    assert first.bounding_box == BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)

    second = result.items[1]
    assert second.text == "world"
    assert second.bounding_box == BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
