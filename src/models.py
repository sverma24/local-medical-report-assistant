from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LabStatus = Literal["low", "normal", "high", "unknown"]


class LabMeasurement(BaseModel):
    test_name: str
    value: float
    unit: str = ""
    reference_text: str = ""
    ref_low: float | None = None
    ref_high: float | None = None
    status: LabStatus = "unknown"
    severity: float = 0.0


class AnalysisOutput(BaseModel):
    summary: str = ""
    areas_of_concern: list[str] = Field(default_factory=list)
    doctor_followup: list[str] = Field(default_factory=list)
    nutritional_signals: list[str] = Field(default_factory=list)
    lifestyle_recommendations: list[str] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(default_factory=list)
    disclaimer: str = ""

