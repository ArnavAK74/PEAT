# PEAT – Protein Engineering Agent Toolkit

> RAG-powered structural biology assistant for the KhareLab.

**Live site**: (https://arnavak74.github.io/PEAT/)

## What it does

PEAT combines three tools in a single Streamlit interface:

1. **Structure analysis** — PDB lookup, 3D viewer, UniProt domain maps, hotspot detection
2. **Literature RAG** — retrieves open-access papers via Unpaywall, answers your questions with an LLM
3. **HPC execution** — runs short GROMACS jobs on your cluster directly from the sidebar

## Quick start

```bash
git clone https://github.com/KhareLab/PEAT.git
cd PEAT
pip install -r requirements.txt
```

Create a `.env` file (or use Streamlit secrets):

```env
# LLM — OpenRouter in dev, point at HPC for prod
OPENROUTER_API_KEY=your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=mistralai/mistral-7b-instruct:free

# RAG
UNPAYWALL_EMAIL=your@email.edu
SCIHUB_ENABLED=false        # set true for dev fallback
LIBRARY_COOKIE=             # paste your library session cookie for prod auth

# HPC (optional)
HPC_HOST=cluster.university.edu
HPC_USER=yournetid
HPC_KEY_PATH=~/.ssh/id_rsa
HPC_WORKDIR=~/peat_runs
```

```bash
streamlit run app.py
```

## LLM routing

| Environment | Setting | Notes |
|-------------|---------|-------|
| Dev | `LLM_BASE_URL=https://openrouter.ai/api/v1` | Free-tier models available |
| Prod | `LLM_BASE_URL=http://your-hpc:8000/v1` | vLLM / Ollama on HPC node |

## RAG routing

PEAT tries paper retrieval in this order:
1. **Unpaywall** — open-access PDF (always on)
2. **Library auth cookie** — set `LIBRARY_COOKIE` env var (production)
3. **Sci-Hub** — set `SCIHUB_ENABLED=true` (dev only)

## HPC commands

Allowed prefixes: `gmx`, `python`, `bash`, `squeue`, `sacct`, `sbatch`, `scancel`, `echo`, `ls`, `cat`, `head`, `tail`

Timeout: 5 minutes (configurable via `HPC_TIMEOUT_SECONDS`).

Example:
```
gmx mdrun -deffnm minimize
```

## Roadmap

See [PEAT Q1 Roadmap](https://kharelab.github.io/PEAT/#roadmap) or the [PEAT_Roadmap.pdf](./PEAT_Roadmap.pdf).

## License

MIT — see [LICENSE](LICENSE).
