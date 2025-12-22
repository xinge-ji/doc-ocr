from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from app.services.ocr.preprocess import OcrPreprocessor


class DummyOrientationModel:
    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, image_path: str, batch_size: int = 1):  # noqa: ARG002
        class Res:
            def __init__(self, label: str) -> None:
                self.label_names = [label]

        return [Res(self.label)]


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_preprocess_rotates_and_saves_with_orientation(tmp_path: Path) -> None:
    src = tmp_path / "input.jpg"
    _make_image(src)

    preprocessor = OcrPreprocessor(
        output_dir=tmp_path / "out",
        orientation_model=DummyOrientationModel("180"),
    )

    result = preprocessor.preprocess(str(src))

    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.angle == 180
    assert page.path.exists()
    # 输出路径应包含 rot180
    assert "rot180" in page.path.name


def test_preprocess_uses_custom_output_dir_and_cleanup(tmp_path: Path) -> None:
    src = tmp_path / "input2.jpg"
    _make_image(src)

    preprocessor = OcrPreprocessor(
        output_dir=tmp_path / "persist",
        orientation_model=DummyOrientationModel("90_degree"),
    )

    result = preprocessor.preprocess(str(src))

    assert result.paths
    for p in result.paths:
        assert p.is_file()
        assert str(p).startswith(str(tmp_path / "persist"))

    # 清理临时目录防止污染
    shutil.rmtree(tmp_path / "persist")
