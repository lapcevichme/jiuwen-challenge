import math
import threading
from dataclasses import dataclass, field

TENTHS = 10


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Reservation:
    amount: int
    settled: bool = field(default=False, init=False)


class BudgetLedger:
    def __init__(self, budget_tokens: float, *, safety_fraction: float = 0.05):
        if not math.isfinite(budget_tokens) or budget_tokens < 0:
            raise ValueError("budget must be finite and non-negative")
        if not 0 <= safety_fraction < 1:
            raise ValueError("safety fraction must be in [0, 1)")
        self.limit = int(budget_tokens * TENTHS * (1 - safety_fraction))
        self.spent = 0
        self.reserved = 0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.unknown_usage = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        with self._lock:
            return self.limit - self.spent - self.reserved

    def reserve(self, input_tokens: int, max_output_tokens: int) -> Reservation:
        amount = max(0, int(input_tokens)) + TENTHS * max(0, int(max_output_tokens))
        with self._lock:
            if self.spent + self.reserved + amount > self.limit:
                raise BudgetExhausted(
                    f"need {amount / TENTHS:.0f} tokens, {(self.limit - self.spent - self.reserved) / TENTHS:.0f} left"
                )
            self.reserved += amount
        return Reservation(amount)

    def settle(self, reservation: Reservation, input_tokens: int, output_tokens: int) -> int:
        cost = int(input_tokens) + TENTHS * int(output_tokens)
        with self._lock:
            self._release(reservation)
            self.spent += cost
            self.calls += 1
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
        return cost

    def charge_unknown(self, reservation: Reservation, *, input_only: bool = False, input_tokens: int = 0) -> int:
        with self._lock:
            if reservation.settled:
                return 0
            cost = min(reservation.amount, int(input_tokens)) if input_only else reservation.amount
            self._release(reservation)
            self.spent += cost
            self.calls += 1
            self.unknown_usage += 1
        return cost

    def _release(self, reservation: Reservation) -> None:
        if reservation.settled:
            raise ValueError("reservation already settled")
        reservation.settled = True
        self.reserved -= reservation.amount

    def summary(self) -> str:
        with self._lock:
            return (
                f"TOKENS spent={self.spent / TENTHS:.1f} input={self.input_tokens} output={self.output_tokens} "
                f"calls={self.calls} unknown_usage={self.unknown_usage} limit={self.limit / TENTHS:.0f}"
            )
