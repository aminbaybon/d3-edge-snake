from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent import decide, format_raw, load_model
from policy import greedy_move, plan_move
from render import render_board
from rl_agent import WEIGHTS, dqn_move, load_qnet
from snake import Direction, SnakeGame

DEATH = {
    "wall": "hit a wall",
    "self": "hit its own body",
    "win": "board filled — win",
}

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
STATE: dict = {
    "status": "Ready",
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
POLICIES = ("plan", "dqn", "d3", "greedy")
POLICY_LABEL = {
    "plan": "Search",
    "dqn": "Double DQN",
    "d3": "d3-edge",
    "greedy": "Greedy",
}
_DQN: tuple | None = None


def _certain(move: Direction) -> dict[str, float]:
    return {direction.value: (1.0 if direction is move else 0.0) for direction in Direction}


def ensure_dqn():
    global _DQN
    if _DQN is None:
        _DQN = load_qnet(WEIGHTS)
    return _DQN


def choose_move(game: SnakeGame, policy: str, use_image: bool) -> tuple[Direction, dict[str, float], float, str]:
    started = time.perf_counter()
    if policy == "d3":
        image = render_board(game) if use_image else None
        decision = decide(game, image)
        return decision.move, decision.probabilities, decision.latency_s, format_raw(decision)
    if policy == "greedy":
        move = greedy_move(game)
        probs = _certain(move)
        note = ""
    elif policy == "dqn":
        try:
            net, board, device = ensure_dqn()
        except FileNotFoundError:
            move, probs = plan_move(game)
            note = "DQN weights are missing; search played this move."
            return move, probs, time.perf_counter() - started, note
        if game.size != board:
            move, probs = plan_move(game)
            note = f"DQN was trained for a {board} board; search played this move."
        else:
            move, probs = dqn_move(net, game, device)
            note = ""
    else:
        move, probs = plan_move(game)
        note = ""
    return move, probs, time.perf_counter() - started, note


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
        "heading": game.heading.value,
        "alive": game.alive,
        "death": DEATH.get(game.death, game.death),
        "food": rel,
        "head": f"{hx}, {hy}",
    }


def run_episode(seed: int, size: int, max_steps: int, use_image: bool, policy: str = "plan") -> None:
    global FRAME
    try:
        if policy not in POLICIES:
            policy = "plan"
        with LOCK:
            STATE.update({
                "status": f"Starting with {POLICY_LABEL[policy]}…",
                "running": True,
                "log": [],
                "summary": None,
                "raw": "",
                "policy": policy,
            })
        game = SnakeGame(size=size, seed=seed)
        with LOCK:
            FRAME = encode_png(game)
            STATE["stats"] = snapshot(game)
            STATE["board"] = board_payload(game)
        if policy == "d3":
            load_model()
            with LOCK:
                STATE["status"] = "Model ready. The game is starting."
        history = []
        last_move = ""
        last_p = None
        while game.alive and not STOP.is_set() and (max_steps <= 0 or game.steps < max_steps):
            greedy = game.greedy_hint().value
            move, probs, latency, note = choose_move(game, policy, use_image)
            applied = game.step(move)
            last_move, last_p = applied.value, float(probs.get(applied.value, 0.0))
            png = encode_png(game, last_move=last_move, last_p=last_p)
            line = (
                f"{game.steps}. {applied.value} "
                f"({round(last_p * 100, 1)}%, {round(latency, 2)}s)"
            )
            if not game.alive:
                line += f" — end: {DEATH.get(game.death, game.death)}"
            row = {
                "step": game.steps,
                "move": applied.value,
                "move_fa": applied.value,
                "confidence": round(float(probs.get(applied.value, 0.0)), 4),
                "probs": probs,
                "greedy": greedy,
                "score": game.score,
                "latency_s": round(latency, 3),
                "policy": policy,
            }
            history.append(row)
            print(
                f"{game.steps:02d} {applied.value:6}  p={probs.get(applied.value, 0):.2f}  "
                f"score={game.score}  {latency:.3f}s  {policy}",
                flush=True,
            )
            with LOCK:
                FRAME = png
                status = f"{POLICY_LABEL[policy]} — step {game.steps} — {applied.value}"
                if note:
                    status = f"{status} — {note}"
                STATE.update(
                    {
                        "status": status,
                        "log": (STATE["log"] + [line])[-24:],
                        "probs": probs,
                        "move_fa": applied.value,
                        "confidence": float(probs.get(applied.value, 0.0)),
                        "latency": latency,
                        "raw": note,
                        "stats": snapshot(game),
                        "board": board_payload(game),
                    }
                )
            if policy != "d3":
                time.sleep(0.68)
        if STOP.is_set() and game.alive:
            ending = "stopped"
        elif game.alive and max_steps > 0:
            ending = f"reached the {max_steps}-step cap"
        else:
            ending = DEATH.get(game.death, game.death)
        summary = {
            "seed": seed,
            "size": size,
            "steps": game.steps,
            "score": game.score,
            "length": game.length,
            "alive": game.alive,
            "death": game.death,
            "death_fa": ending,
            "used_image": use_image and policy == "d3",
            "policy": policy,
            "history": history,
        }
        run_dir = ROOT / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (ROOT / "run" / "final.png").write_bytes(FRAME)
        with LOCK:
            STATE.update(
                {
                    "status": f"Result: {ending} — score {game.score} in {game.steps} steps",
                    "running": False,
                    "summary": summary,
                    "board": board_payload(game),
                }
            )
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update({"status": f"Error: {exc}", "running": False, "raw": traceback.format_exc()})


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
        if path.path == "/play.gif":
            self._send(200, (ROOT / "assets" / "play.gif").read_bytes(), "image/gif")
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
                    "policy": STATE.get("policy", "plan"),
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
            requested = q.get("policy", ["plan"])[0]
            args = (
                int(q.get("seed", ["7"])[0]),
                int(q.get("size", ["8"])[0]),
                0,
                q.get("image", ["1"])[0] != "0",
                requested if requested in POLICIES else "plan",
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


def run_headless(seed: int, size: int, max_steps: int, use_image: bool, policy: str) -> dict:
    STOP.clear()
    run_episode(seed, size, max_steps, use_image, policy)
    return STATE.get("summary") or {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Snake played on a grid action space")
    parser.add_argument("--episode", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0, help="0 plays until the board is full or the snake dies")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--policy", choices=POLICIES, default="plan")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    game = SnakeGame(size=args.size, seed=args.seed)
    global FRAME
    FRAME = encode_png(game)
    STATE["stats"] = snapshot(game)
    STATE["board"] = board_payload(game)
    if args.episode:
        summary = run_headless(
            args.seed, args.size, args.max_steps, use_image=not args.no_image, policy=args.policy
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, ensure_ascii=False, indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
