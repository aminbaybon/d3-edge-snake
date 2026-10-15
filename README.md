# Decision Snake

A snake that plays on the grid itself. The default policy is a search: the shortest path to the apple that still leaves a route back to the tail, otherwise the next cell of a Hamiltonian cycle. It moves one cell at a time, only horizontally or vertically, until the board is full or the snake dies.

![A search game in progress](assets/play.gif)

The optional [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) policy looks at a rendered frame and picks a direction. On the recorded 8×8 game (seed 7) it walked right four times and hit the wall. Combining that distribution with Double DQN made the learned policy worse. The measurements are in [REPORT.md](REPORT.md).

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

d3-edge needs `transformers==5.17.0` because the checkpoint is a Qwen3.5 readout. On a GTX 1650 (compute capability 7.5) the runtime is loaded in float16, since that GPU has no bfloat16 CUDA kernels. Each model decision takes a few seconds.
