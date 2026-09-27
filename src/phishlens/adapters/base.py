"""Abstract adapter interface shared by the three branches."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

from .. import config
from ..schemas import STATUS_OK, BranchEvidence

T = TypeVar("T")


def load_with_retry(
    fn: Callable[[], T],
    attempts: int = config.LOAD_ATTEMPTS,
    wait_s: float = config.LOAD_RETRY_WAIT_S,
) -> T:
    """Call fn, retrying on OSError.

    On Windows the Hugging Face cache can fail once with a symlink privilege
    OSError and then succeed on the next attempt.
    """
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except OSError as exc:
            last = exc
            if attempt < attempts:
                time.sleep(wait_s)
    assert last is not None
    raise last


class BaseAdapter(ABC):
    """One evidence branch.

    Subclasses set `name`, `model_id` and `revision`, and implement `_load`
    and `_predict`. `predict` measures latency and fills in the common fields.
    Exceptions are allowed to propagate: the orchestrator isolates them.
    """

    name: str = "base"
    model_id: str = ""
    revision: str = ""

    def __init__(self) -> None:
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.load_ms: Optional[float] = None

    # ---- loading -------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> None:
        """Load the model once. Safe to call from several threads."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001
                self._load_error = "%s: %s" % (type(exc).__name__, exc)
                raise
            self.load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            self._load_error = None
            self._loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Load weights and tokenizer or processor."""

    # ---- inference -----------------------------------------------------
    @abstractmethod
    def _predict(self, value: Any) -> BranchEvidence:
        """Return evidence with probability, truncated and details filled in."""

    def predict(self, value: Any) -> BranchEvidence:
        self.load()
        t0 = time.perf_counter()
        with self._infer_lock:
            evidence = self._predict(value)
        evidence.branch = self.name
        evidence.status = STATUS_OK
        evidence.input_provided = True
        evidence.model_id = self.model_id
        evidence.revision = self.revision
        evidence.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return evidence

    def describe(self) -> dict:
        return {
            "branch": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "loaded": self._loaded,
            "load_ms": self.load_ms,
            "load_error": self._load_error,
        }
