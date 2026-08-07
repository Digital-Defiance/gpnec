# GPNEC technical report (LaTeX)

Sources for the paper *GPNEC: A Metal State Automaton on Apple Silicon*.

## Build

```bash
cd papers/gpnec
latexmk -pdf gpnec.tex
```

## Regenerating measurement transcripts

From the repo root on a Metal Mac:

```bash
swift run gpnec verify 2>&1 | tee papers/gpnec/verify-all.txt
swift run gpnec bench --backends metal,cpu,cpu-mt --sizes 32,64,128,256 \
  --steps 50 --warmup 64 2>&1 | tee papers/gpnec/bench.txt
swift run gpnec route-verify --n 800 --k 10 --crash 250 --packets 512 \
  --pre 80 --post 120 --seed 42 --betweenness-samples 64 --policy both \
  2>&1 | tee papers/gpnec/route-verify-paired.txt
```

Publication crash control is **`--policy symmetric`** (identical retry+respawn).
Sandbox is UI narrative only; use `--policy both` when updating paper tables.
