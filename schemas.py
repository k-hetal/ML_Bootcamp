from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

UNSPECIFIED = "UNSPECIFIED"
_EMPTY = {"", "unspecified", "unknown", "n/a", "na", "none", "tbd", "not specified", "not stated", "null"}
STATEMENT_TYPES = ["Decision", "Proposal", "Action", "Request", "Unresolved"]
TASK_STATUSES = ["confirmed", "proposed", "needs_review"]


def _norm_unspecified(v):
    if v is None or str(v).strip().lower() in _EMPTY:
        return UNSPECIFIED
    return str(v).strip()


def fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "--:--"
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ActionItem(BaseModel):
    task: str = Field(description="Specific work to be done.")
    owner: str = Field(default=UNSPECIFIED)
    deadline: str = Field(default=UNSPECIFIED)
    status: str = Field(default="needs_review", description="confirmed | proposed | needs_review")
    review_note: str = Field(default="", description="Why a human should check this item (empty if none).")
    evidence: str = ""
    start: Optional[float] = None
    end: Optional[float] = None

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _fix(cls, v):
        return _norm_unspecified(v)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in TASK_STATUSES else "needs_review"

    @property
    def label(self) -> str:
        if self.status == "proposed":
            return "Proposed, not yet agreed"
        if self.status == "needs_review":
            return "Needs human review"
        if self.owner == UNSPECIFIED or self.deadline == UNSPECIFIED:
            return "Confirmed, owner/deadline unspecified"
        return "Confirmed commitment"


class Decision(BaseModel):
    decision: str
    evidence: str = ""
    start: Optional[float] = None
    end: Optional[float] = None


class Statement(BaseModel):
    text: str
    type: str = "Proposal"
    start: Optional[float] = None
    segment: Optional[int] = None

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        v = str(v or "").strip().capitalize()
        return v if v in STATEMENT_TYPES else "Proposal"


class Topic(BaseModel):
    title: str
    start: float = 0.0
    end: float = 0.0
    start_segment: int = 0
    end_segment: int = 0


class MeetingRecord(BaseModel):
    summary: str
    minutes: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    statements: List[Statement] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)


class Evidence(BaseModel):
    quote: str
    start: float
    end: float


class Answer(BaseModel):
    question: str
    answer: str
    evidence: List[Evidence] = Field(default_factory=list)
    grounded: bool = True
