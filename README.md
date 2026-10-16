# Decision Snake

This is a snake game, and a record of how we chose the player that runs it.

The clip is that player. It moves one cell at a time, only up, down, left, or right, and it continues until the board is full or the snake dies.

![Search playing snake, one cell at a time](assets/play.gif)

## What we set out to do

The snake was playing badly. We wanted it to choose the best move it could. The plan was to show the rendered board to a vision model, then combine that model with reinforcement learning so the snake would improve. The clip above is the result of testing that plan. The model is still in the repository. It is no longer the player.

## The model

We started with [d3-edge](https://huggingface.co/vllm-sr/d3-edge), a decision model of about 0.8 billion parameters built on Qwen3.5. Each turn we drew the board as a picture and asked the model which direction to take.

On the game we recorded, an 8×8 board with seed 7, the apple was above the head. The useful move was up. The model chose right four times and hit the wall on step 4. The score was 0. Each look took about three seconds, and its confidence in "right" rose as the wall got closer. Left was absent from every decision.

The picture was a poor input. The game already stores the position of every cell. The image is a drawing of those positions, so the model had to reconstruct a state the program already held.

We ran this on a GTX 1650 with 4 GB of memory. That GPU cannot run the model's native bfloat16 kernels, so the weights were loaded in float16. After loading, the model occupied 1.11 GB.

## Reinforcement learning

A snake has three legal moves: turn left, go straight, or turn right. Reversing direction is illegal. For a small set of discrete moves, Double DQN is the method that fits. It keeps every step it sees and learns from it again. Policy-gradient methods made for language models discard a step after one update. Algorithms made for continuous control, such as SAC, answer a different kind of problem.

Training the 0.8B model itself was out of reach on this GPU. At three seconds per decision, 200,000 training steps would take about 167 hours of inference before a single learning update, and the activations for training do not fit in 4 GB. The decision layer is a readout over labeled options, so there is no policy to update from the game's score. The model stayed frozen.

We kept its recorded preferences and used them as a teacher in two ways.

1. During training, the reward of each step was shifted toward the direction the model preferred.
2. At decision time, the model's preference was added to the network's own score, and the snake took the move with the highest total.

Both versions played worse than the network alone. The network trained with the model's reward averaged a score of 0.4 and died immediately. Adding the preference at decision time left the snake alive for 250 steps with a score of about 0.3, because the recorded model had given left turns almost no weight. With left turns removed, the apple stays out of reach.

Double DQN on its own did learn. Over 200,000 steps, on 30 random 8×8 games, its average score was 15.9. A plain rule that steps toward the apple when the next cell is safe scored 18.3. Both are ordinary. Both still die.

Two earlier training runs never got that far, and the reason is worth keeping. One reward punished eating: a new apple usually appears farther away, so the snake lost points for the action we wanted. Another showed the network the whole board as a flat list of cells, which does not say whether the next move dies. After the inputs were reduced to a few facts about the three legal moves — whether each one is blocked, where the apple sits relative to the head, and how long the snake is — the same network reached a score of about 15 within 25,000 steps. The learning method had been fine. The reward and the inputs had been wrong.

## The player in the clip

The default player searches the grid.

It takes the shortest path to the apple only when the snake can follow that entire path and still reach its own tail after eating. The tail check is what keeps it out of a dead end. When the path to the apple fails that test, and the board side is even, it follows a fixed loop that visits every cell, again only when the tail stays reachable. When that step is also unsafe, it moves toward its tail. The last resort is the step that leaves the most open space.

The loop without the tail check was lethal: 29 of 30 snakes died by step 800. With the check, deaths fell to zero.

On 30 random 8×8 games, allowed to continue for up to 2000 steps, search scored 56 on average. The highest possible score is 61. The median game filled the board. Fifteen of the thirty games won. None died. The other fifteen were still alive when the step limit of that measurement stopped them. The live page has no step limit.

| Player | Average score | What happened |
| --- | ---: | --- |
| d3-edge, the recorded game | 0 | Hit the wall on step 4 |
| DQN trained toward the model | 0.4 | Died immediately |
| DQN plus the model's preference | 0.3 | Stayed alive and barely ate |
| Double DQN alone | 16 | Learned a middling policy, then usually died |
| Step toward the apple | 18 | Simple, and usually fatal |
| Search, up to 2000 steps | 56 | No deaths. 15 of 30 boards filled |

## Where that leaves the model

The model was unnecessary for this game. The board is already known exactly. Passing a drawing of it through a 0.8B model produced a slow and confident mistake, and teaching a smaller network to follow that mistake lowered the score from about 16 to about 0. The snake in the clip is the search described above. Press Start on the local page and that is the policy that plays.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Search is selected by default. Double DQN, the greedy rule, and d3-edge are in the same menu. d3-edge needs `transformers==5.17.0` and takes a few seconds per move.

One game from the terminal, with no step limit:

```powershell
.\.venv\Scripts\python app.py --episode --policy plan --seed 7 --size 8
```
