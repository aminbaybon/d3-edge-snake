# Decision Snake

Search plays the grid. The 0.8B vision model was measured, including as a teacher for Double DQN, and it made play worse.

![Search playing snake, one cell at a time](assets/play.gif)

Run date: 15 October 2026. Hardware: NVIDIA GeForce GTX 1650, 4 GB, compute capability 7.5, PyTorch 2.14.1+cu126.

## Result

The vision model was not needed. Using its action distribution as a teacher made Double DQN worse.

The rendered frame is a deterministic function of the grid. The action space is three relative moves: turn left, go straight, turn right. A 180-degree reversal is already illegal. On an 8×8 board the best decision procedure is search over that grid, not a 0.8B model looking at pixels.

On seed 7, size 8:

- d3-edge chose right four times and hit the wall on step 4. Score 0. Each decision took about 3.0–3.2 seconds.
- The same seed with search, under the 400-step cap that the page used at the time, scored 39 and was still alive.

The page no longer has a step cap. Search now plays until the board is full or the snake dies. The animation above is that search.

## The model

[d3-edge](https://huggingface.co/vllm-sr/d3-edge) is a decision readout on Qwen3.5-0.8B (about 594 million parameters, BF16 weights). This GPU has no BF16 CUDA kernels, so the runtime was patched to load the backbone in float16. After loading, 1.11 GiB of GPU memory was allocated. `flash-linear-attention` and `causal_conv1d` were not installed, so the PyTorch reference kernels ran, which is why a decision takes about three seconds.

All four recorded decisions had the head facing right and the apple above the head. The greedy hint was up every time. The model saw that and chose right, and the probability of right rose as the wall got closer:

| Step | up | down | right | left |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.287 | 0.227 | 0.486 | absent |
| 2 | 0.314 | 0.260 | 0.426 | absent |
| 3 | 0.274 | 0.180 | 0.546 | absent |
| 4 | 0.274 | 0.169 | 0.557 | absent |

Left never appeared in the distribution. This is one timed episode, not a 30-seed average. Those four decisions were enough to build the teacher used below: when the food was above the head, the model walked into the right wall. The episode was recorded while the board labels were still Persian. The page and the prompts are English now; the numbers below are from that recorded run.

## Action space

The right mapping from an observation to an action is not to give the pixels to the model.

The true state is discrete and fully observed: body cells, head, apple, and heading. The picture is that state plus display noise. Reading a decision back out of pixels throws away coordinates the simulator already has.

The action space that fits snake is the three relative moves. Four absolute headings minus the reverse heading are exactly those three moves. Both the network and the search decide in that space.

The features on which Double DQN actually learned are aligned with those moves:

- three danger bits: whether left, straight, and right would die immediately
- the current heading, one-hot
- the sign of the apple offset on each axis
- the apple in the head's frame, forward and to the right, divided by the board size
- length divided by the area

An MLP on the flattened grid did not learn this. Details are below.

## Why the 0.8B model was not fine-tuned

Double DQN is the suitable algorithm for this game. Updating the language model is not.

- The action set is small and discrete. DQN reuses each transition. PPO and GRPO discard it after one update and need more samples on this environment.
- SAC is for continuous actions.
- One d3-edge decision takes about 3 seconds. 200,000 DQN steps at that rate are about 167 hours of inference, before any backward pass.
- The float16 weights already occupy 1.11 GiB. Training activations for the vision tower and the linear-attention layers do not fit in 4 GB.
- The decision head is a readout over labeled options, not a differentiable policy that can be updated from the snake reward.

The model stayed frozen. Only its recorded action distribution was passed in as a teacher. That is the combination that could actually be measured on this GPU.

## The two combinations that were run

The teacher distribution is the mean of those four decisions. Left had no mass, so it was given a numerical floor of `1e-6` and the distribution was renormalized. The log of that floor is about -13.8.

1. Reward bonus. Each step's reward gained `0.5 * log π_d3(absolute direction)`. Action selection stayed epsilon-greedy, so the teacher did not freeze the behavior policy from the first update. 200,000 steps, the same budget as DQN alone.

2. Decision-time mixture, on the DQN that had not seen the teacher: `argmax (Q(s, a) + 1.0 * log π_d3(a))`.

## Numbers

Board 8×8. The maximum score is 61 (64 cells minus the starting length of 3). Unless noted, 30 seeds and a 250-step cap.

| Policy | Mean score | Median | Max | Mean steps | Death rate | Full-board rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d3-edge (seed 7 only) | 0 | 0 | 0 | 4 | 1 | 0 |
| DQN trained with the d3 reward | 0.40 | 0 | 2 | 4.5 | 1.00 | 0 |
| DQN plus the d3 prior at decision time | 0.30 | 0 | 1 | 250 | 0 | 0 |
| Double DQN alone | 15.93 | 16.5 | 21 | 118.6 | 0.97 | 0 |
| Greedy (nearest safe cell) | 18.27 | 18 | 30 | 120.2 | 1.00 | 0 |
| Search, 250-step cap | 30.7 | 30 | 36 | 250 | 0 | 0 |
| Search, 2000-step cap | 56.0 | 61 | 61 | until cap or win | 0 | 15 of 30 |

Search is still alive at the 250-step cap; the cap cuts the game off. At 2000 steps the mean is 56, the median is 61 (a full board), the minimum is 37, and there are no deaths. Fifteen seeds filled the board. The other fifteen were still alive at step 2000 and had stalled in the endgame, not crashed.

The decision-time mixture shows a death rate of zero together with a score of 0.3: the snake survives 250 steps and almost never eats. The absolute-right prior removes left turns from the action space, because the learned Q values do not differ by 13 units. Without that turn the apple is effectively out of reach.

Training curves (greedy evaluation every 25,000 steps, 20 seeds, 120-step cap):

- DQN alone locked onto a score of about 15 by step 25,000 and stayed there through 200,000 (15.15, 14.2, 14.2, 14.9, 15.15, 14.35, 15.0, 15.75). Death rate stayed between 0.50 and 0.80.
- DQN with the d3 reward died immediately: score 0.30 to 0.45 and a death rate of 1 on all eight evaluations.

DQN beats d3-edge and matches the greedy heuristic. It stays well behind search, and more training did not close that gap.

## Two training runs that were discarded

These were a bad reward and a bad observation, not the final result.

- The potential `γΦ(s') − Φ(s)` with `Φ = −Manhattan distance`, including the step that eats the apple. The next apple is usually farther away, so eating scores about −6 and cancels the +1 apple reward. On a flattened grid the mean scores from 25k to 100k steps were 2.2, 3.1, 3.0, 3.3. The death rate rose from 0.70 to 0.90.
- The corrected reward (apple +1, death −1, and a 0.05 step toward the food only between apples) still with an MLP on the 192-dimensional grid. Through 75k steps the scores were 2.10, 2.15, 2.40.

After the observation was changed to the action-aligned features, the same network reached a score of 15 by step 25,000. The difference was not the algorithm. The raw grid does not tell the network whether the next action dies.

## The search that plays now

The page default is `plan`.

1. Shortest path to the apple, accepted only when the snake can follow every cell and can still reach its tail after eating. That rejects dead-end coils.
2. If that path is unsafe and the board has an even side, the next cell of a fixed Hamiltonian cycle, and only if the tail is still reachable after that step.
3. Otherwise chase the tail, and finally take the move that opens the most free space.

The cycle fallback without the tail check killed the snake (29 deaths out of 30 by step 800). With the check, deaths dropped to zero.

This search does not learn and is not connected to d3-edge. On a small fully observed board it is a better policy than a small network that has seen 200,000 transitions.

## Was the model necessary?

No.

The exact state is already in the simulator. The model is a multimodal classifier that maps a rendered frame to a distribution over labels. On the recorded episode that distribution leaned into the right wall, assigned nothing to left, and became more confident as the wall approached. Feeding that distribution into DQN, either as a reward or as an addition to Q, took the score from about 16 back to about 0.

What plays the game well is search on the grid. Double DQN on the relative action space learns a middling policy, comparable to the greedy heuristic, and is not required at this board size. d3-edge did not add a policy, a teacher, or a feature that the search did not already have.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Choose Search, Double DQN, Greedy, or d3-edge. Search does not use a step cap.

One headless game, also with no step cap:

```powershell
.\.venv\Scripts\python app.py --episode --policy plan --seed 7 --size 8
```

d3-edge needs `transformers==5.17.0` because the checkpoint is a Qwen3.5 readout. On a GTX 1650 the runtime is loaded in float16. Each model decision takes a few seconds.

The same write-up is in [REPORT.md](REPORT.md).
