from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from PIL import Image

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


def build_state(game: SnakeGame) -> dict[str, Any]:
    hx, hy = game.head
    fx, fy = game.food
    _, _, rel = game.relative_food()
    return {
        "game": "Snake",
        "goal": "Eat the apple. Do not hit a wall or the snake's body. A 180-degree reversal is illegal.",
        "board_size": game.size,
        "coordinates": "Column 0 is the left edge and increases to the right. Row 0 is the top edge and increases downward.",
        "head": {"column": hx, "row": hy},
        "body": [{"column": x, "row": y} for x, y in game.snake],
        "apple": {"column": fx, "row": fy},
        "heading_label": game.heading.value,
        "heading": game.heading.value,
        "apple_relative_to_head": rel,
        "score": game.score,
        "length": game.length,
        "step": game.steps,
        "legal_move_labels": [d.value for d in game.legal_moves()],
        "legal_moves": [d.value for d in game.legal_moves()],
        "ascii_board": game.ascii_board(),
        "image_reminder": "In the image the head is yellow, the body is green, the apple is red, and the labels are English.",
    }


def build_questions(game: SnakeGame) -> dict[str, Any]:
    criteria = {}
    for move in game.legal_moves():
        nx, ny = game.next_cell(move)
        criteria[move.value] = (
            f"{move.value} — the head moves to column {nx} row {ny}"
        )
    return {
        "move": {
            "type": "choice",
            "instructions": (
                "Looking at the board image and the JSON state, "
                "which move should the snake make now so it gets closer to the apple "
                "without hitting a wall or its own body? Choose exactly one option."
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


def decide(game: SnakeGame, board_image: Image.Image | None) -> Decision:
    import time

    model = load_model()
    state = build_state(game)
    questions = build_questions(game)
    kwargs: dict[str, Any] = {"state": state, "questions": questions}
    if board_image is not None:
        kwargs["images"] = [board_image.convert("RGB")]
    started = time.perf_counter()
    result = model.system_one(**kwargs)
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
        used_image=board_image is not None,
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
