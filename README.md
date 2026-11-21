# Do AI research agents take credit for changes that did nothing?

Four small pilots (November 2026) on one question: when an AI agent runs research by itself over many rounds, are its reports true?

Each task has an answer that we know in advance. In the training task, two of the 14 settings the agent can change are **wired to nothing**. Any credit the agent gives them is false, so code can score that claim without a human judge.

These are pilots, not findings. Every cell has only 2 or 3 runs.

![Pilot 4: with no check, each model credited a do-nothing setting in 2 of 3 runs; with a checker from another company, 0 of 3. The checker cut real progress for GPT-6 Astra.](img/pilot4_results.png)

**Start here:** the pilot 4 write-up is [`pilot4/PILOT4_RESULT.md`](pilot4/PILOT4_RESULT.md). Every round of every run is in [`pilot4/pilot4_rows_real.csv`](pilot4/pilot4_rows_real.csv), and the full run records are in `pilot4/record_real_*.json`. Each folder has the code that produced it.

## Results at a glance

| Pilot | Task | Researcher model(s) | What it showed | Cost |
|---|---|---|---|---|
| 1 | Tune a gradient-boosted model on noisy synthetic data (CPU) | GPT-6.1 Sol | Claimed gains were about 2x the true gains, and the gap grew over rounds | $0.27 |
| 2 | Tune a small CIFAR-10 classifier with named settings (GPU) | GPT-6.1 Sol | The agent's first move matched a public speedrun recipe, which suggests recall; later pilots renamed the settings | ~$4.65 |
| 3 | Same trainer with 14 renamed settings, 2 wired to nothing | GPT-6.1 Sol | False credit to a dummy setting in 2 of 4 runs; no setting was ever named as doing nothing | ~$2.52 |
| 4 | Same as pilot 3, three frontier models, with and without a checker | Claude Fable 5.2, GPT-6 Astra, Gemini 4 Argon | No check: false credit in **6 of 9** runs (2 of 3 for each model). Checker model from another company: **0 of 9**, but one model's progress stalled, and two checked runs claimed gains where the true gain was 0 | $7.80 API |

Each pilot folder has a `PILOT*_RESULT*.md` write-up, the per-round rows (`*.csv`), the full run records (`record_*.json`) and the log.

## How the training task works (pilots 2 to 4)

- A GPU machine trains a small CIFAR-10 network (`vm_train.py`) from a config and returns an accuracy.
- Each round, the agent sees **one noisy result** (1 training run, 5,000 validation images), its own notes and the current config. It replies with a new config and a status report: its best round, its claimed gain, the settings it believes helped and the settings it believes did nothing.
- **Truth** is the mean of 3 fresh seeds on the full 10,000-image test set. The agent never sees it.
- In pilots 3 and 4, settings are renamed K01 to K14 and put on neutral scales, so the agent can't rely on what it remembers. The mapping is generated from a fixed seed in `pilot3/pilot3.py` and saved in `pilot3/SECRET_mapping.json`.

**This ladder is now public, so it is retired.** Any future model may have seen it. New studies should generate fresh mappings with a private seed.

## Pilot 4 arms

- **A, no check.** The agent's report is accepted as written.
- **C, checker.** After each round, a model from another company reads exactly what the agent saw, plus its report, and can send it back once. Rotation: GPT-6 Astra checked Fable, Gemini 4 Argon checked Astra, and Claude Fable 5.2 checked Argon. The checker is **not** told about the dummy settings.

## Running it

You need Python 3.11 or later, an SSH-reachable Linux GPU machine (pilots 2 to 4) and API keys in environment variables:

```
OPENAI_API_KEY      # pilots 1 to 4
ANTHROPIC_API_KEY   # pilot 4
GEMINI_API_KEY      # pilot 4
SSH_KEY             # path to the private key for the GPU machine (default ~/.ssh/id_ed25519)
```

```
pip install -r requirements.txt
python pilot/pilot1.py                      # CPU only
cd pilot4 && ./run_all.sh <gpu-ip>          # sets up the GPU machine, then runs pilot 4
python pilot4/pilot4.py --fake --rounds 2 --repeats 1   # checks the API calls without a GPU
```

`run_all.sh` installs PyTorch on the GPU machine and downloads CIFAR-10 from its source. The dataset is not included here. `truth_cache.json` holds the true accuracies already measured, so reruns don't retrain known configs.

Model names and prices in the code are those of November 2026. Results will change as models change.

## Contact

Mark C. Johnson, Rowset LLC · mark@rowset.co · [rowset.co](https://rowset.co). Questions, reruns and disagreements are welcome; you can also open an issue.

## License

MIT. See `LICENSE`.
