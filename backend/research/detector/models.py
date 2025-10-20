import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Char(BaseModel):
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.0
    text: str = ""
    bbox_valid: bool = False
    bbox: list[float] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("polygon", mode="before")
    @classmethod
    def _validate_polygon(
        cls, value: Iterable[Sequence[float]] | None
    ) -> list[list[float]]:
        return _ensure_polygon(value or [])

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, value: Iterable[float] | None) -> list[float]:
        return [float(coord) for coord in value or []]


class TextLine(BaseModel):
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.0
    text: str = ""
    chars: list[Char] = Field(default_factory=list)
    original_text_good: bool = False
    words: list[str] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("polygon", mode="before")
    @classmethod
    def _validate_polygon(
        cls, value: Iterable[Sequence[float]] | None
    ) -> list[list[float]]:
        return _ensure_polygon(value or [])

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, value: Iterable[float] | None) -> list[float]:
        return [float(coord) for coord in value or []]

    @field_validator("words", mode="before")
    @classmethod
    def _ensure_words(cls, value: Iterable[object] | None) -> list[str]:
        return [str(word) for word in value or []]


class Page(BaseModel):
    text_lines: list[TextLine] = Field(default_factory=list)
    image_bbox: list[float] = Field(default_factory=list)
    page: int = 0

    model_config = ConfigDict(extra="ignore")

    @field_validator("image_bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, value: Iterable[float] | None) -> list[float]:
        return [float(coord) for coord in value or []]


class OCRResult(BaseModel):
    pages: list[Page] = Field(default_factory=list)
    success: bool = True
    error: str = ""
    page_count: int = 0
    status: str = "unknown"

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _populate_page_count(self) -> "OCRResult":
        if not self.page_count:
            self.page_count = len(self.pages)
        return self

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OCRResult":
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, json_file_path: str | Path) -> "OCRResult":
        with open(json_file_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def _ensure_polygon(vertices: Iterable[Sequence[float]]) -> list[list[float]]:
    return [list(map(float, vertex)) for vertex in vertices]
