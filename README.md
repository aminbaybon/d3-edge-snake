# Decision Snake

[d3-edge](https://huggingface.co/vllm-sr/d3-edge) came out on 10 October 2026. I gave it a snake game and wrote down what each setup did.

![The winning game, every move](assets/play.gif)

The clip is the run that filled the board: 8×8, seed 7, score 61. Every move is in it, played back faster than the live page.

## What I tried, and what happened

| Setup | Result |
| --- | --- |
| Picture of the board | Score 0. Four turns right, into the wall |
| Coordinate matrix | Turned toward the apple and ate it |
| Body listed as cells | Clean opening, then score 2 at step 8 |
| Path since the last apple, and what each move does | Score 9 in 135 steps, model alone |
| Network trained to follow the model's preference | Average 0.4, and 0.3 when the preference was added at decision time |
| Network on its own | Average 16 |
| Step toward the apple when the next cell is safe | Average 18 |
| Search, 30 boards, up to 2000 steps | Average 56. 15 boards filled |
| Network takes over when the path repeats | Scores 18, then 13, then 31 |
| Model while the path is new; network only if a closer step still reaches the tail; search for the rest of that chase | Score 61 in 978 steps. Board filled |

In that last game the model ate 23 apples, including the last two. The network ate 12. Search ate 26.

A model look in that game took 1.3 seconds. On the same positions, search took 53 microseconds a move and the network 121.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Start uses d3-edge plus the network, with search only for the chase above. Needs `transformers==5.17.0`.

```powershell
.\.venv\Scripts\python app.py --episode --policy guided --seed 7 --size 8
```
