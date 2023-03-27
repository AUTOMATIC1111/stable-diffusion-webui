"""Pydantic data models and other dataclasses. This is the only file that uses Optional[]
typing syntax instead of | None syntax to work with pydantic"""
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class PredictBody(BaseModel):
    session_hash: Optional[str]
    event_id: Optional[str]
    data: List[Any]
    event_data: Optional[Any]
    fn_index: Optional[int]
    batched: Optional[
        bool
    ] = False  # Whether the data is a batch of samples (i.e. called from the queue if batch=True) or a single sample (i.e. called from the UI)
    request: Optional[
        Union[Dict, List[Dict]]
    ] = None  # dictionary of request headers, query parameters, url, etc. (used to to pass in request for queuing)


class ResetBody(BaseModel):
    session_hash: str
    fn_index: int


class InterfaceTypes(Enum):
    STANDARD = auto()
    INPUT_ONLY = auto()
    OUTPUT_ONLY = auto()
    UNIFIED = auto()


class Estimation(BaseModel):
    msg: Optional[str] = "estimation"
    rank: Optional[int] = None
    queue_size: int
    avg_event_process_time: Optional[float]
    avg_event_concurrent_process_time: Optional[float]
    rank_eta: Optional[float] = None
    queue_eta: float


class ProgressUnit(BaseModel):
    index: Optional[int]
    length: Optional[int]
    unit: Optional[str]
    progress: Optional[float]
    desc: Optional[str]


class Progress(BaseModel):
    msg: str = "progress"
    progress_data: List[ProgressUnit] = []
