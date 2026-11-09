from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    instruction: str
    reward: int | float
    task_dir: Path
    workspace: Path | None = None
    check: str | None = None


class SolveStatus(StrEnum):
    SOLVED = "SOLVED"
    UNSOLVABLE = "UNSOLVABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    answer: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolveStatus):
            raise ValueError("invalid solve status")
        if self.status is SolveStatus.SOLVED:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("a solved task needs a non-empty answer")
        elif self.answer is not None or not self.detail.strip():
            raise ValueError("an unresolved task needs a reason and no answer")

    @classmethod
    def solved(cls, answer: str, detail: str = "") -> "SolveResult":
        return cls(SolveStatus.SOLVED, answer=answer.strip(), detail=detail)

    @classmethod
    def unsolvable(cls, detail: str) -> "SolveResult":
        return cls(SolveStatus.UNSOLVABLE, detail=detail)

    @classmethod
    def unsupported(cls, detail: str) -> "SolveResult":
        return cls(SolveStatus.UNSUPPORTED, detail=detail)

    @classmethod
    def error(cls, detail: str) -> "SolveResult":
        return cls(SolveStatus.ERROR, detail=detail)


Solver = Callable[[TaskSpec], SolveResult]
