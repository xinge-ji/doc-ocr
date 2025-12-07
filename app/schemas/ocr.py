from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left/top x coordinate")
    y1: float = Field(..., description="Left/top y coordinate")
    x2: float = Field(..., description="Right/bottom x coordinate")
    y2: float = Field(..., description="Right/bottom y coordinate")

    def as_xyxy(self) -> list[float]:
        """Return the bounding box as a simple [x1, y1, x2, y2] list."""

        return [self.x1, self.y1, self.x2, self.y2]


class OcrItem(BaseModel):
    text: str
    bounding_box: BoundingBox
    page: int = Field(default=1, ge=1)
    block_id: int | None = None
    line_id: int | None = None


class OcrResult(BaseModel):
    items: list[OcrItem] = Field(default_factory=list)
