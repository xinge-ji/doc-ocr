import httpx
import pytest

from app.schemas.ocr import BoundingBox
from app.services.ocr.hunyuan_ocr import HunyuanOcrClient


class FakePreprocessor:
    class _Page:
        def __init__(self, path: str, page: int) -> None:
            self.path = path
            self.page = page

    class _Result:
        def __init__(self, pages) -> None:
            self.pages = pages

    def __init__(self, path: str) -> None:
        self.path = path

    def preprocess(self, *_args, **_kwargs):
        return self._Result([self._Page(self.path, 1)])


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_hunyuan_ocr_client_parses_bbox_and_text():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "foo(1,2),(3,4)bar(10,20),(30,40)",
                }
            }
        ]
    }
    fake_client = FakeHttpClient(payload)
    client = HunyuanOcrClient(
        base_url="http://localhost:8000/v1",
        api_key="KEY",
        model="Tencent-Hunyuan/HunyuanOCR",
        preprocessor=FakePreprocessor("dummy.jpg"),
        http_client=fake_client,
    )

    result = await client.extract(b"dummy-bytes")

    assert len(result.items) == 2
    first, second = result.items

    assert first.text == "foo"
    assert first.bounding_box == BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
    assert first.page == 1

    assert second.text == "bar"
    assert second.bounding_box == BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)

    assert fake_client.calls
    call = fake_client.calls[0]
    assert call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer KEY"
    payload = call["json"]
    assert payload["messages"][1]["content"][0]["type"] == "image_url"
    assert payload["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image")
