from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from snake import Direction, SnakeGame, parse_direction

MODEL_ID = "vllm-sr/d3-edge"

_model = None
_load_error = ""


@dataclass
class Decision:
    move: Direction
    probabilities: dict[str, float]
    raw: dict[str, Any]
    latency_s: float
    used_image: bool

    @property
    def confidence(self) -> float:
        return float(self.probabilities.get(self.move.value, 0.0))


def _distance(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def move_text(game: SnakeGame, move: Direction, trail: list[tuple[int, int]] | None) -> str:
    """What one option does, including whether it steps back onto the recent path."""
    nx, ny = game.next_cell(move)
    hx, hy = game.head
    fx, fy = game.food
    before = _distance(hx, hy, fx, fy)
    after = _distance(nx, ny, fx, fy)
    if after < before:
        distance = f"closer to the apple, distance {before} becomes {after}"
    elif after > before:
        distance = f"farther from the apple, distance {before} becomes {after}"
    else:
        distance = f"the same distance from the apple, {before}"
    if not (0 <= nx < game.size and 0 <= ny < game.size):
        place = "outside the board, a wall"
    elif game.would_die(move) == "self":
        place = "the snake's body"
    elif (nx, ny) == game.food:
        place = "the apple"
    else:
        tail = game.snake[0]
        seen = (nx, ny) in set(trail or []) and (nx, ny) != tail
        if seen:
            place = "empty, but the head already left this cell since the last apple"
        else:
            place = "empty, and not on the path since the last apple"
    return f"{move.value} — column {nx}, row {ny}; {place}; {distance}"


def coordinate_state(game: SnakeGame, trail: list[tuple[int, int]] | None = None) -> str:
    """The board as text. Column 0 is the left edge. Row 0 is the top edge."""
    head = game.head
    food = game.food
    body = set(game.snake)
    columns = " ".join(f"{x:>2}" for x in range(game.size))
    lines = [
        "Snake. Decide from these coordinates. There is no picture.",
        "Column 0 is the left edge and increases to the right. Row 0 is the top edge and increases downward.",
        "Matrix codes: . empty, H head, B body, A apple.",
        f"columns: {columns}",
    ]
    for y in range(game.size):
        cells = []
        for x in range(game.size):
            pos = (x, y)
            if pos == head:
                cells.append(" H")
            elif pos == food:
                cells.append(" A")
            elif pos in body:
                cells.append(" B")
            else:
                cells.append(" .")
        lines.append(f"row {y}:{''.join(cells)}")
    hx, hy = head
    fx, fy = food
    lines.append(f"Head: column {hx}, row {hy}. Facing {game.heading.value}.")
    lines.append(f"Apple: column {fx}, row {fy}.")
    lines.append("Snake body as coordinates, head first. Each pair is column, row:")
    for index, (x, y) in enumerate(reversed(game.snake), start=1):
        role = "head" if index == 1 else ("tail" if index == len(game.snake) else "body")
        lines.append(f"{index}. {role}: column {x}, row {y}")
    lines.append(f"Score {game.score}. Length {game.length}. Step {game.steps}.")
    lines.append("Path of the head since the last apple, oldest first. Each pair is column, row:")
    walked = list(trail or [])
    if len(walked) <= 1:
        lines.append("The head has not left a trail yet.")
    else:
        for index, (x, y) in enumerate(walked, start=1):
            lines.append(f"{index}. column {x}, row {y}")
    lines.append(
        "A move onto that path repeats a cell the head already left. "
        "Prefer an empty cell that is closer to the apple and is not on that path."
    )
    return "\n".join(lines)


def build_questions(game: SnakeGame, trail: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    criteria = {move.value: move_text(game, move, trail) for move in game.legal_moves()}
    return {
        "move": {
            "type": "choice",
            "instructions": (
                "Using the coordinates, the path since the last apple, and the result of each option, "
                "choose the move that gets closer to the apple. "
                "Do not enter a wall or the body. "
                "Do not return to a cell already on that path when another empty cell is closer to the apple."
            ),
            "criteria": criteria,
        }
    }


def _as_probs(blob: Any) -> dict[str, float]:
    if isinstance(blob, dict):
        nested = blob.get("probs") or blob.get("probabilities") or blob.get("distribution") or blob.get("p")
        if isinstance(nested, dict):
            blob = nested
        out: dict[str, float] = {}
        for key, value in blob.items():
            if isinstance(value, dict) and "prob" in value:
                value = value["prob"]
            if isinstance(value, (int, float)):
                parsed = parse_direction(key)
                out[(parsed.value if parsed else str(key))] = float(value)
        return out
    if isinstance(blob, list):
        out = {}
        for item in blob:
            if isinstance(item, dict):
                label = item.get("id") or item.get("value") or item.get("label")
                score = item.get("prob") or item.get("p") or item.get("score")
                parsed = parse_direction(label)
                if parsed is not None and isinstance(score, (int, float)):
                    out[parsed.value] = float(score)
        return out
    return {}


def _choice_from_answer(answer: Any, legal: list[Direction]) -> tuple[Direction | None, dict[str, float]]:
    probs = _as_probs(answer)
    picked: Direction | None = None
    if isinstance(answer, dict):
        for key in ("value", "answer", "choice", "id", "label", "selected"):
            if key in answer:
                picked = parse_direction(answer[key])
                break
    elif isinstance(answer, str):
        picked = parse_direction(answer)
    if picked is None and probs:
        picked = parse_direction(max(probs, key=probs.get))
    if picked is not None and picked not in legal:
        if probs:
            ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            for name, _ in ordered:
                cand = parse_direction(name)
                if cand in legal:
                    picked = cand
                    break
        else:
            picked = None
    return picked, probs


def _force_fp16_runtime(root: str) -> None:
    """GTX 16-series cannot run bfloat16. The runtime hardcodes that dtype."""
    import os

    path = os.path.join(root, "d3_runtime.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    needle = "dtype=torch.bfloat16,"
    if needle not in text:
        return
    text = text.replace(needle, "dtype=torch.float16,", 1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("Patched d3 runtime to float16 for this GPU.", flush=True)


def _patch_broken_scientific_stack() -> None:
    """This machine's conda scipy/sklearn cannot import on NumPy 2; d3-edge does not need them."""
    import sys
    import types
    from importlib.machinery import ModuleSpec

    import numpy as np
    import transformers.utils as utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
        np.Infinity = np.inf
        np.NINF = -np.inf

    def fake(name: str, is_pkg: bool = False):
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, loader=None, is_package=is_pkg)
        module.__file__ = "<stub>"
        if is_pkg:
            module.__path__ = []
        sys.modules[name] = module
        return module

    for key in list(sys.modules):
        if key == "scipy" or key.startswith("scipy.") or key == "sklearn" or key.startswith("sklearn."):
            del sys.modules[key]

    scipy = fake("scipy", True)
    optimize = fake("scipy.optimize", True)
    optimize.linear_sum_assignment = lambda *args, **kwargs: None
    scipy.optimize = optimize

    def _no_sklearn() -> bool:
        return False

    utils.is_sklearn_available = _no_sklearn
    import_utils.is_sklearn_available = _no_sklearn
    import_utils._is_sklearn_available = False


def load_model():
    global _model, _load_error
    if _model is not None:
        return _model
    if _load_error:
        raise RuntimeError(_load_error)
    try:
        import os
        import sys

        import torch
        import transformers  # noqa: F401
        from huggingface_hub import snapshot_download

        _patch_broken_scientific_stack()

        if torch.cuda.is_available():
            device = "cuda:0"
            name = torch.cuda.get_device_name(0)
            major, _minor = torch.cuda.get_device_capability(0)
            # Turing and older have no bfloat16 CUDA kernels. d3-edge ships BF16.
            if major < 8:
                os.environ["D3_EDGE_FORCE_FP16"] = "1"
            print(f"Loading {MODEL_ID} on {name} ({device})...", flush=True)
        else:
            device = "cpu"
            threads = max(1, (os.cpu_count() or 4) // 2)
            torch.set_num_threads(threads)
            print(f"Loading {MODEL_ID} on CPU with {threads} threads...", flush=True)
        root = snapshot_download(MODEL_ID)
        if os.environ.get("D3_EDGE_FORCE_FP16") == "1":
            _force_fp16_runtime(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from d3_runtime import D3

        # The fp16 edit changes d3_runtime.py, so the manifest hash no longer matches.
        verify = "none" if os.environ.get("D3_EDGE_FORCE_FP16") == "1" else "fast"
        _model = D3.from_pretrained(root, device=device, verify=verify)
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"Model ready on GPU ({allocated:.2f} GiB allocated).", flush=True)
        else:
            print("Model ready.", flush=True)
        return _model
    except Exception as exc:
        _load_error = str(exc)
        raise


def decide(game: SnakeGame, trail: list[tuple[int, int]] | None = None) -> Decision:
    import time

    model = load_model()
    started = time.perf_counter()
    result = model.system_one(
        state=coordinate_state(game, trail),
        questions=build_questions(game, trail),
    )
    latency = time.perf_counter() - started
    answers = result.get("answers", result) if isinstance(result, dict) else {}
    move_answer = answers.get("move", answers) if isinstance(answers, dict) else answers
    picked, probs = _choice_from_answer(move_answer, game.legal_moves())
    if picked is None:
        picked = game.heading
    return Decision(
        move=picked,
        probabilities=probs,
        raw=result if isinstance(result, dict) else {"result": result},
        latency_s=latency,
        used_image=False,
    )


def format_raw(decision: Decision) -> str:
    payload = {
        "move": decision.move.value,
        "label": decision.move.value,
        "probabilities": decision.probabilities,
        "latency_s": round(decision.latency_s, 3),
        "answers": decision.raw.get("answers", decision.raw),
        "usage": decision.raw.get("usage"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
