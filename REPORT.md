# Decision Snake

This game gives the downloaded [d3-edge](https://huggingface.co/vllm-sr/d3-edge) model the board as a coordinate matrix, not as a picture. A small network, trained by reinforcement learning, edits that decision when the model would die or when another safe move is clearly better.

The clip shows the motion: one cell at a time, only up, down, left, or right, until the board is full or the snake dies. Press Start on the local page and the model is what proposes the move.

![The snake moving one cell at a time](assets/play.gif)

## What we set out to do

The point is to see this model on a real control problem and improve its decision with reinforcement learning. d3-edge reads a text matrix of the cells and proposes a direction. The learned network then keeps that proposal, or replaces it.

## The model

d3-edge is a decision model of about 0.8 billion parameters, built on Qwen3.5. It can answer from text alone, so each turn it receives a matrix: row 0 is the top, column 0 is the left, and every cell is empty, head, body, or apple. An earlier run sent a picture of the same 8×8 game, seed 7. The apple was above the head, and from the picture the model chose right four times into the wall. From the coordinate matrix, on that same opening, its own first choice was up (probability 0.52), the next choice was up again, and it ate the apple. After four moves the score was 1 and the snake was still alive. Left, which the picture run never offered, appeared in the later text decisions. A text decision takes about two seconds.

The test GPU was a GTX 1650 with 4 GB. It has no bfloat16 CUDA kernels, so the weights were loaded in float16 and occupied 1.11 GB.

## Reinforcement learning

At each step the snake has three legal moves: turn left, go straight, or turn right. Reversing is illegal. Double DQN fits that choice. It keeps every step and learns from it again.

Training the 0.8B weights themselves does not fit in 4 GB, and at three seconds per look a training run of 200,000 steps would be about 167 hours of inference before a learning update. The decision layer is a readout over labels. The weights stay frozen.

We first tried to improve the network by making it copy the model. That failed, and the failure is why the live player does not copy. Shifting the training reward toward the model's recorded preference dropped the average score from about 16 to 0.4. Adding that same preference on top of the network's score at decision time left the snake alive and it almost never ate, because the recorded model had given left turns almost no weight.

Trained on its own, Double DQN reached an average score of about 16, close to a plain rule that steps toward the apple when the next cell is safe. The network learned once its inputs said whether each of the three moves dies and where the apple sits. An earlier reward had punished eating, and an earlier input was the whole board as a flat list. Those two runs stayed near a score of 2 or 3.

## How the two are combined

The player behind Start asks the downloaded model for a fresh proposal from the current coordinate matrix. The learned network then edits it.

- If the model's move would hit a wall or the body, it is refused.
- If another safe move is clearly more valuable, the network plays that instead.
- Otherwise the model's move is the one that is played.

On the recorded game the raw model said right and died. With the editor, that right turn is refused. The pads on the page still show what the model itself answered.

Search is in the menu as a comparison. On 30 random 8×8 games it scored 56 on average, filled 15 boards, and never died. It never calls the model. It is not the default.

| Player | Average score | What happened |
| --- | ---: | --- |
| d3-edge alone, the recorded game | 0 | Hit the wall on step 4 |
| DQN trained to copy the model | 0.4 | Died immediately |
| DQN plus the model's preference | 0.3 | Stayed alive and barely ate |
| Double DQN alone | 16 | Learned a middling policy, then usually died |
| Step toward the apple | 18 | Simple, and usually fatal |
| Search, up to 2000 steps | 56 | No deaths. 15 of 30 boards filled |

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The selected policy is d3-edge plus Double DQN. The model takes a few seconds per move and needs `transformers==5.17.0`.

One game from the terminal:

```powershell
.\.venv\Scripts\python app.py --episode --policy guided --seed 7 --size 8 --max-steps 8
```
