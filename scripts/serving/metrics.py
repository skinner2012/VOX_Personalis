"""In-memory SLA metrics collection and JSONL persistence."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path


class MetricsCollector:
    """Collects per-segment latency and failure metrics.

    Two storage layers:
    - In-memory: P50/P95/min/max/mean latency, failure rate (resets on restart)
    - JSONL file: append-only per-segment log that survives restarts
    """

    def __init__(self, metrics_out: Path) -> None:
        self._metrics_out = metrics_out
        self._latencies: list[float] = []
        self._failure_count: int = 0
        self._total_segments: int = 0
        self._total_audio_sec: float = 0.0
        self._start_time: float = time.monotonic()
        self._segment_counter: int = 0

    def record_segment(
        self,
        *,
        segment_id: int,
        latency_ms: float,
        duration_sec: float,
        status: str,
    ) -> None:
        """Record one completed segment (ok or failed). Appends JSONL entry."""
        self._total_segments += 1
        self._total_audio_sec += duration_sec

        if status == "ok":
            self._latencies.append(latency_ms)
        else:
            self._failure_count += 1

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "segment_id": segment_id,
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(latency_ms, 1),
            "status": status,
            "model_version": "v2",
        }
        self._append_jsonl(entry)

    def next_segment_id(self) -> int:
        """Return and increment the segment counter."""
        self._segment_counter += 1
        return self._segment_counter

    def get_metrics(self) -> dict:
        """Return current in-memory metrics as a dict (JSON-serialisable)."""
        uptime = time.monotonic() - self._start_time
        failed = self._failure_count
        total = self._total_segments
        failure_rate = round(failed / total, 4) if total > 0 else 0.0

        latency_stats: dict = {}
        if self._latencies:
            import statistics

            sorted_lat = sorted(self._latencies)
            n = len(sorted_lat)
            p50_idx = int(n * 0.50)
            p95_idx = min(int(n * 0.95), n - 1)
            latency_stats = {
                "p50": round(sorted_lat[p50_idx], 1),
                "p95": round(sorted_lat[p95_idx], 1),
                "min": round(sorted_lat[0], 1),
                "max": round(sorted_lat[-1], 1),
                "mean": round(statistics.mean(self._latencies), 1),
            }
        else:
            latency_stats = {"p50": None, "p95": None, "min": None, "max": None, "mean": None}

        return {
            "uptime_sec": round(uptime, 1),
            "total_segments": total,
            "failed_segments": failed,
            "failure_rate": failure_rate,
            "latency_ms": latency_stats,
            "total_audio_sec": round(self._total_audio_sec, 1),
            "model_version": "v2",
        }

    def _append_jsonl(self, entry: dict) -> None:
        """Append one JSON line to the metrics file. Creates parent dirs if needed."""
        try:
            self._metrics_out.parent.mkdir(parents=True, exist_ok=True)
            with self._metrics_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            # Never let a metrics write failure crash the serving path
            pass
