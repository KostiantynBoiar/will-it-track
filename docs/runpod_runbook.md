# RunPod runbook — SAM 3 inference → scores → modelling (over SSH)

The pod version of `docs/colab_runbook.md`: a real machine you drive over SSH, with a persistent
**network volume** at `/workspace` (replaces Google Drive). No sleep, no tab, no session cap — long runs
survive in `tmux`.

## 0. Create the pod (RunPod UI, one-time)
- **Template:** `madiator2011/better-pytorch:cuda12.4-torch2.6.0` (torch 2.6 / CUDA 12.4 — fine for SAM 3).
- **GPU:** RTX 4090 / A5000 / A40 / A100 (24 GB is plenty; must be **Ampere+ or Ada** for `bf16` — avoid T4).
- **Network Volume:** attach one, mounted at **`/workspace`** → your `outputs/` persist, so runs are resumable.
- **SSH key:** RunPod → *Settings → SSH Public Keys* → paste your `~/.ssh/id_ed25519.pub`.
- Start the pod, then copy the exact SSH command it shows (either `ssh root@<ip> -p <port> -i <key>` or the
  `ssh <pod-id>@ssh.runpod.io -i <key>` proxy form).

## 1. SSH in + set the environment
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519            # use RunPod's exact command
cd /workspace

export HF_TOKEN=hf_xxx                                   # your granted HF token
export GITHUB_PAT=ghp_xxx                                # to clone the private repo
export HF_HUB_DISABLE_XET=1                              # avoid the Xet 403 on fine-grained tokens
export SAFARI_PATHS__DATA_ROOT=/workspace/data           # frames + annotations on the persistent volume
export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs     # predictions + parquets (resumable)
```

## 2. Clone + install
```bash
cd /workspace
git clone https://$GITHUB_PAT@github.com/KostiantynBoiar/will-it-track.git
cd will-it-track
pip -q install -r requirements-local.txt                 # torch already present → not reinstalled
pip -q install -U "transformers>=5.0" accelerate         # ships Sam3VideoModel
# transformers can pull a torch built for a CUDA newer than the pod driver → pin torch to the driver's
# CUDA (12.8 here). If a run later dies with "NVIDIA driver too old", this is the fix:
pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3 third_party/sam3   # the VEval scorer (not the model)
pip -q install iopath einops timm ftfy regex             # scorer + its (undeclared) model-import deps
#   (Colab preinstalls these; a bare pod doesn't. If score.py later reports a missing module, pip-install it.)
```

## 3. Guard — assert the stack before any GPU work
```bash
python -c "import torch, transformers, numpy; print('torch', torch.__version__, torch.version.cuda, '| tf', transformers.__version__, '| np', numpy.__version__); from transformers import Sam3VideoModel, Sam3VideoProcessor; print('SAM 3 video OK')"
nvidia-smi -L
```
If the import fails: `pip -q install -U "git+https://github.com/huggingface/transformers"` and retry.

## 4. Fetch the gated annotations
```bash
# $HF_TOKEN in the env is enough (acquire + from_pretrained both read it). Optional explicit login:
hf auth login --token $HF_TOKEN            # note: `huggingface-cli` is deprecated → use `hf`
python -m src.acquire --annotations        # ~0.9 GB → /workspace/data/annotations
```

## 5. Smoke test — 3 probes end-to-end
```bash
python -m src.inference.harness --split test --limit 3
python -m src.eval.score --split test                     # ~15 min: VEval scores the whole test GT
python -c "import os,pandas as pd; print(pd.read_parquet(os.environ['SAFARI_PATHS__OUTPUTS_ROOT']+'/scores.parquet').dropna(subset=['pDetA']).head())"
```

## 6. Full test split — in tmux (survives an SSH disconnect)
```bash
tmux new -s run
python -m src.inference.harness --split test && python -m src.eval.score --split test
#   Ctrl-b then d  to detach;  tmux attach -t run  to come back;  the run keeps going either way.
```
~3 h on an A100 (proportionally longer on a 4090). Resumable: re-run the same line after any interruption.

## 7. Modelling (H2 / Gate 2)
```bash
python -m src.features.assemble        # -> outputs/features.parquet (pulls train-site frames on demand)
python -m src.analysis.regression      # -> outputs/models/{pDetA,pAssA}_beta.pkl (+ coef CSVs)
python -m src.analysis.variance        # dominance + VIF + the det-vs-assoc contrast
python -m src.analysis.cross_val       # leave-location-out OOS error -> outputs/validation/
python -m src.analysis.uncertainty     # -> outputs/figures/predictive_line_{det,assoc}.png
```

## 8. Get the results back to your laptop
```bash
# from your laptop:
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<ip>:/workspace/outputs ./outputs
#   or use `runpodctl send` / the RunPod file browser for outputs/*.parquet + outputs/figures/
```

## Notes
- **Persistence:** everything under `/workspace` is on the network volume → survives disconnects and pod
  restarts, so inference resumes where it stopped. The `outputs/predictions/*.json` are the durable
  source of truth.
- **Re-install after a full STOP:** pip packages land on the *container* disk, not the volume. If you stop
  (not just disconnect) and restart the pod, re-run step 2's `pip` lines (~3 min); your data/outputs stay.
- **GPU tier:** `bf16` is the default (needs Ampere+/Ada). On anything without bf16, `export
  SAFARI_INFERENCE__PRECISION=fp16`. `bitsandbytes` is not needed.
- **Speed:** frame downloads are 16-way parallel to the pod's fast filesystem (no Drive penalty), so
  expect a much better per-probe rate than Colab-on-Drive.
- **Robustness / species split:** re-run steps 6–7 with `export SAFARI_INFERENCE__PROMPT_MODE=generic`
  (generic prompt) and `--split train` (for the H1 species hold-out) when you're ready.
