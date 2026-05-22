from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator


class CustomerFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

    @validator("age")
    def age_range(cls, v):
        if not 18 <= v <= 95:
            raise ValueError("age must be between 18 and 95")
        return v

    @validator("duration")
    def duration_nonnegative(cls, v):
        if v < 0:
            raise ValueError("duration must be >= 0")
        return v

    @validator("campaign")
    def campaign_positive(cls, v):
        if v < 1:
            raise ValueError("campaign must be positive")
        return v

    @validator("pdays")
    def pdays_rule(cls, v):
        if v != -1 and v < 0:
            raise ValueError("pdays must be -1 or >= 0")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(ge=0.0, le=1.0)
    threshold_decision: str
    model_version: str
    latency_ms: float = Field(ge=0.0)


class BatchScoreItem(BaseModel):
    id: int
    conversion_band: str
    probability: float = Field(ge=0.0, le=1.0)


class BatchScoreRequest(BaseModel):
    customers: Optional[List[CustomerFeatures]] = None
    csv_path: Optional[str] = None


class BatchScoreResponse(BaseModel):
    results: List[BatchScoreItem]


class MetricsResponse(BaseModel):
    latency_p50: float
    latency_p99: float
    request_count: int
    error_count: int
    prediction_distribution: dict
