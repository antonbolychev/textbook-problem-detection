from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProblemLineEntry(BaseModel):
    index: int
    text: str
    matches_problem_text: bool | None = None

    model_config = ConfigDict(extra="ignore")


class ProblemReport(BaseModel):
    problem_id: str
    line_indices: list[int] = Field(default_factory=list)
    lines: list[ProblemLineEntry] = Field(default_factory=list)
    full_text: str = ""
    bbox: dict[str, float] | None = None  # {"x": float, "y": float, "width": float, "height": float}

    model_config = ConfigDict(extra="ignore")


class LineReport(BaseModel):
    index: int
    text: str
    assigned_problem_id: str | None = None
    belongs_to_problem: bool = False
    matches_problem_text: bool | None = None

    model_config = ConfigDict(extra="ignore")


class PageReport(BaseModel):
    page: int
    problems: list[ProblemReport] = Field(default_factory=list)
    lines: list[LineReport] = Field(default_factory=list)
    unassigned_lines: list[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")
