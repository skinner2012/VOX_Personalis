"""
Gemma 4 GGUF subprocess wrapper — M4.

Spawns llama-cli as a persistent child process so that the model weights stay
loaded across corrections. Each correct() call writes the prompt to stdin and
reads the generated response, using the performance-stats line emitted by
llama-cli as the generation-complete sentinel.

The startup banner (ASCII art + model info + available commands) is drained
once at construction time via a quiet-period heuristic. Subsequent calls
assume the child is idle and ready for the next prompt.

Why persistent (no --single-turn): even with mmap'd GGUF the first prompt is
the slowest and process startup is non-trivial. Restarting per correction
would make the 361-clip batch eval impractical and the live demo unusable.

Why single-line prompt: in llama-cli's default conversation mode, every `\n`
in stdin terminates the current user turn and triggers a generation. Embedding
multi-line prompts (e.g. "Correct…\n\nInput: …\nOutput:") splits one logical
prompt into three separate turns, which is what caused the off-by-one Stage B
responses in the first eval iteration. Keeping the entire prompt on a single
line is the simplest reliable fix. `--no-conversation` was investigated and
discarded — this llama-cli build prints
"--no-conversation is not supported by llama-cli, please use llama-completion
instead" and silently falls back to conversation mode.
"""

import queue
import subprocess
import threading
import time

GEMMA_MODEL_DEFAULT = "./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
LLAMA_CLI_DEFAULT = "/Users/skinnercheng/llama.cpp/build/bin/llama-cli"

PROMPT_TEMPLATE = (
    "You are lightly polishing a live ASR transcript from a backend/infrastructure "
    "engineer answering technical hiring-manager interview questions. Make MINIMAL "
    "changes: fix capitalization, punctuation, and only OBVIOUS typos. Preserve the "
    "original wording unless it is clearly wrong. Never invent words or add information. "
    "Output only the corrected text: {stage_a_text}"
)

# Minimum word-level overlap (input∩output / input) required to accept Gemma's
# rewrite. Below this we treat the output as a hallucination and pass Stage A
# through unchanged. > 50% catches the gross cases ("directions" -> "Elike
# sound by.", 0% overlap; "i so agree" -> "I had a feeling...", 0.33% on
# stop-words only; "set the" -> "Please provide the transcript...", exactly
# 50% via "the") while accepting noisy-but-helpful rewrites ("play my phone
# phone" -> "Play my phone.", 100%; "we are telephonic" -> "We are on a
# telephone call.", 67%).
_MIN_INPUT_OVERLAP = 0.51

# llama-cli prints this line at the end of every generation turn.
_STATS_SENTINEL = "[ Prompt:"

# llama-cli prints this in response to the /clear interactive command.
_CLEAR_SENTINEL = "Chat history cleared"

# Lines that are part of the startup banner or system output — not corrections.
_SKIP_PREFIXES = (
    ">",
    "[",
    "build",
    "model ",
    "modali",
    "Loading",
    "available",
    "/exit",
    "/regen",
    "/clear",
    "/read",
    "/glob",
    "Exiting",
)


def _is_safe_correction(original: str, corrected: str) -> bool:
    """Reject corrections that share too few words with the input.

    Compares lowercase alphanumeric word sets and returns False when the
    correction retains less than _MIN_INPUT_OVERLAP of the original's distinct
    words — the signature of a Gemma hallucination unrelated to the input.
    """

    def words(s: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in s)
        return set(cleaned.split())

    orig = words(original)
    if not orig:
        return True
    overlap = len(orig & words(corrected)) / len(orig)
    return overlap >= _MIN_INPUT_OVERLAP


class GemmaWorker:
    """Persistent llama-cli subprocess for Gemma 4 transcript corrections."""

    def __init__(
        self,
        model_path: str = GEMMA_MODEL_DEFAULT,
        llama_cli: str = LLAMA_CLI_DEFAULT,
        n_threads: int = 8,
        n_tokens: int = 60,
        startup_timeout: float = 120.0,
    ) -> None:
        self.proc = subprocess.Popen(
            [
                llama_cli,
                "-m",
                model_path,
                "--reasoning",
                "off",
                "-n",
                str(n_tokens),
                "--simple-io",
                "-t",
                str(n_threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self._drain_startup(startup_timeout)

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, ""):
            self._q.put(line)

    def _drain_startup(self, timeout: float) -> None:
        # Drain the initial burst of banner + model-load output.
        # Readiness signal: 1.5s quiet window after the last line of output.
        deadline = time.monotonic() + timeout
        last_output_at = time.monotonic()
        seen_any = False
        while time.monotonic() < deadline:
            try:
                self._q.get(timeout=0.5)
                last_output_at = time.monotonic()
                seen_any = True
            except queue.Empty:
                if seen_any and (time.monotonic() - last_output_at) > 1.5:
                    return
        # Timeout reached — proceed anyway; the model may be ready.

    def correct(self, stage_a_text: str, timeout: float = 3.0) -> str:
        """Return Gemma-corrected text, or stage_a_text on timeout/error.

        Issues llama-cli's /clear command before every correction so each call
        runs against a fresh conversation context. Without this, after ~300
        prompts the accumulated chat history causes Gemma to hallucinate
        responses from earlier inputs — verified in the M4 first eval, where
        damage was 18× higher in the last 60 clips than in the first 100.
        """
        assert self.proc.stdin is not None
        self._drain_queue()
        self.proc.stdin.write("/clear\n")
        self.proc.stdin.flush()
        self._wait_for_clear()

        prompt = PROMPT_TEMPLATE.format(stage_a_text=stage_a_text)
        self.proc.stdin.write(prompt + "\n")
        self.proc.stdin.flush()

        deadline = time.monotonic() + timeout
        collected: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=0.05)
                if _STATS_SENTINEL in line:
                    break
                collected.append(line)
            except queue.Empty:
                pass

        extracted = self._extract(collected, stage_a_text)
        if extracted != stage_a_text and not _is_safe_correction(stage_a_text, extracted):
            return stage_a_text
        return extracted

    def _drain_queue(self) -> None:
        # Drain any leftover output (trailing blanks, "> " indicators from the
        # previous turn) so the next correction starts from a clean stream.
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _wait_for_clear(self, timeout: float = 1.0) -> None:
        # Block until llama-cli echoes "Chat history cleared." then drain any
        # trailing blank line / next prompt indicator before returning.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=0.05)
                if _CLEAR_SENTINEL in line:
                    # Short follow-up drain for the trailing blank/> indicator
                    drain_deadline = time.monotonic() + 0.15
                    while time.monotonic() < drain_deadline:
                        try:
                            self._q.get(timeout=0.05)
                        except queue.Empty:
                            pass
                    return
            except queue.Empty:
                pass
        # Timed out — proceed anyway; worst case the next prompt still works
        # but starts with the previous context still in place.

    def _extract(self, lines: list[str], fallback: str) -> str:
        # If the prompt was echoed by the model, find the last "Output:" line
        # and take everything after it. If not echoed (no-TTY subprocess),
        # take all lines that are not system/banner output.
        output_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped == "output:" or stripped.startswith("output: "):
                output_idx = i

        candidate_lines = lines[output_idx + 1 :] if output_idx >= 0 else lines

        result: list[str] = []
        for ln in candidate_lines:
            s = ln.strip()
            if not s:
                continue
            if any(s.startswith(p) for p in _SKIP_PREFIXES):
                continue
            result.append(s)

        text = " ".join(result).strip()
        return text if text else fallback

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self) -> "GemmaWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
