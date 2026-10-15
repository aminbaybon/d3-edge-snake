from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent import decide, format_raw, load_model
from render import render_board
from snake import FA_DIR, SnakeGame, to_fa_digits

DEATH_FA = {
    "wall": "برخورد با دیوار",
    "self": "برخورد با بدن خودش",
    "win": "کل صفحه پر شد — برد",
}

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
STATE: dict = {
    "status": "آماده",
    "log": [],
    "running": False,
    "summary": None,
    "probs": {},
    "move_fa": "",
    "confidence": 0.0,
    "latency": 0.0,
    "raw": "",
    "stats": {},
    "board": None,
}
FRAME: bytes = b""
STOP = threading.Event()


def encode_png(game: SnakeGame, last_move: str = "", last_p: float | None = None) -> bytes:
    buf = io.BytesIO()
    render_board(game, last_move=last_move, last_p=last_p).save(buf, format="PNG")
    return buf.getvalue()


def board_payload(game: SnakeGame) -> dict:
    return {
        "size": game.size,
        "snake": [list(p) for p in game.snake],
        "food": list(game.food),
        "heading": game.heading.value,
        "alive": game.alive,
        "death": game.death,
        "score": game.score,
        "steps": game.steps,
        "length": game.length,
    }


def snapshot(game: SnakeGame) -> dict:
    hx, hy = game.head
    _, _, rel = game.relative_food()
    return {
        "score": game.score,
        "steps": game.steps,
        "length": game.length,
        "heading": FA_DIR[game.heading],
        "alive": game.alive,
        "death": DEATH_FA.get(game.death, game.death),
        "food": rel,
        "head": f"{to_fa_digits(hx)}، {to_fa_digits(hy)}",
    }


def run_episode(seed: int, size: int, max_steps: int, use_image: bool) -> None:
    global FRAME
    try:
        with LOCK:
            STATE.update({"status": "بارگذاری مدل d3-edge…", "running": True, "log": [], "summary": None, "raw": ""})
        game = SnakeGame(size=size, seed=seed)
        with LOCK:
            FRAME = encode_png(game)
            STATE["stats"] = snapshot(game)
            STATE["board"] = board_payload(game)
        load_model()
        with LOCK:
            STATE["status"] = "مدل آماده شد. بازی شروع می‌شود."
        history = []
        last_move = ""
        last_p = None
        while game.alive and game.steps < max_steps and not STOP.is_set():
            image = render_board(game)
            greedy = FA_DIR[game.greedy_hint()]
            decision = decide(game, image if use_image else None)
            applied = game.step(decision.move)
            last_move, last_p = FA_DIR[applied], decision.confidence
            png = encode_png(game, last_move=last_move, last_p=last_p)
            line = (
                f"{to_fa_digits(game.steps)}. {FA_DIR[applied]} "
                f"({to_fa_digits(round(decision.confidence * 100, 1))}٪، "
                f"{to_fa_digits(round(decision.latency_s, 2))} ث)"
            )
            if not game.alive:
                line += f" — پایان: {DEATH_FA.get(game.death, game.death)}"
            row = {
                "step": game.steps,
                "move": applied.value,
                "move_fa": FA_DIR[applied],
                "confidence": round(decision.confidence, 4),
                "probs": decision.probabilities,
                "greedy": greedy,
                "score": game.score,
                "latency_s": round(decision.latency_s, 3),
            }
            history.append(row)
            print(
                f"{game.steps:02d} {applied.value:6}  p={decision.confidence:.2f}  "
                f"score={game.score}  {decision.latency_s:.1f}s",
                flush=True,
            )
            with LOCK:
                FRAME = png
                STATE.update(
                    {
                        "status": f"قدم {to_fa_digits(game.steps)} — {FA_DIR[applied]}",
                        "log": STATE["log"] + [line],
                        "probs": decision.probabilities,
                        "move_fa": FA_DIR[applied],
                        "confidence": decision.confidence,
                        "latency": decision.latency_s,
                        "raw": format_raw(decision),
                        "stats": snapshot(game),
                        "board": board_payload(game),
                    }
                )
        if STOP.is_set() and game.alive:
            ending = "متوقف شد"
        elif game.alive:
            ending = f"به سقف {to_fa_digits(max_steps)} قدم رسید"
        else:
            ending = DEATH_FA.get(game.death, game.death)
        summary = {
            "seed": seed,
            "size": size,
            "steps": game.steps,
            "score": game.score,
            "length": game.length,
            "alive": game.alive,
            "death": game.death,
            "death_fa": ending,
            "used_image": use_image,
            "history": history,
        }
        run_dir = ROOT / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (ROOT / "run" / "final.png").write_bytes(FRAME)
        with LOCK:
            STATE.update(
                {
                    "status": f"نتیجه: {ending} — امتیاز {to_fa_digits(game.score)} از {to_fa_digits(game.steps)} قدم",
                    "running": False,
                    "summary": summary,
                    "board": board_payload(game),
                }
            )
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update({"status": f"خطا: {exc}", "running": False, "raw": traceback.format_exc()})


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path)
        if path.path in ("/", "/index.html"):
            self._send(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path.path == "/board.png":
            with LOCK:
                body = FRAME
            self._send(200, body, "image/png")
            return
        if path.path == "/api/state":
            with LOCK:
                payload = {
                    "status": STATE["status"],
                    "log": STATE["log"],
                    "running": STATE["running"],
                    "probs": STATE["probs"],
                    "move_fa": STATE["move_fa"],
                    "confidence": STATE["confidence"],
                    "latency": STATE["latency"],
                    "board": STATE["board"],
                    "stats": STATE["stats"],
                    "summary": None
                    if STATE["summary"] is None
                    else {k: v for k, v in STATE["summary"].items() if k != "history"},
                }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path.path == "/api/start":
            q = parse_qs(path.query)
            if STATE["running"]:
                self._send(200, b'{"ok":false}', "application/json")
                return
            STOP.clear()
            args = (
                int(q.get("seed", ["7"])[0]),
                int(q.get("size", ["8"])[0]),
                int(q.get("max_steps", ["25"])[0]),
                q.get("image", ["1"])[0] != "0",
            )
            threading.Thread(target=run_episode, args=args, daemon=True).start()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path.path == "/api/stop":
            STOP.set()
            self._send(200, b'{"ok":true}', "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args: object) -> None:
        if args and str(args[0]).startswith("GET /board.png"):
            return
        super().log_message(fmt, *args)


def run_headless(seed: int, size: int, max_steps: int, use_image: bool) -> dict:
    STOP.clear()
    run_episode(seed, size, max_steps, use_image)
    return STATE.get("summary") or {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Snake played by vllm-sr/d3-edge")
    parser.add_argument("--episode", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    game = SnakeGame(size=args.size, seed=args.seed)
    global FRAME
    FRAME = encode_png(game)
    STATE["stats"] = snapshot(game)
    STATE["board"] = board_payload(game)
    if args.episode:
        summary = run_headless(args.seed, args.size, args.max_steps, use_image=not args.no_image)
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, ensure_ascii=False, indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"باز کن: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
