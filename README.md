# PEAT – Protein Engineering Agent Toolkit

> RAG-powered structural biology assistant for the KhareLab at Rutgers University.

**Live app**: http://149.165.172.252

## What it does

PEAT is a single Streamlit chatbot that combines four capabilities:

1. **Structure analysis** — PDB lookup, interactive 3D viewer (3Dmol.js), UniProt domain maps, M-CSA active sites, AlphaFold structure fetching
2. **Literature RAG** — retrieves open-access papers via Unpaywall, runs LLM Q&A against the full text
3. **Structural similarity** — Foldseek search against PDB, AlphaFold DB, and SwissProt
4. **HPC execution** — submits GROMACS energy minimization jobs to Anvil (NAIRR allocation) via SSH, monitors job status, and downloads results — all from the chat box

## Usage

Everything goes through the chat box:

| Input | What happens |
|-------|-------------|
| `Analyze 6B5X` or just `6B5X` | Full PDB + UniProt + literature analysis |
| FASTA block or raw AA sequence | BLAST → top PDB match → analysis |
| `alphafold Q9WYE2` | Fetch AlphaFold predicted structure |
| `foldseek 6B5X` | Structural similarity search |
| `minimize 6B5X` | Submit GROMACS energy minimization on Anvil |
| `check job 1234567` | Poll SLURM job status |
| `download results 1234567` | Pull output files from Anvil |
| `gmx ...` / `sbatch ...` / `squeue` | Run arbitrary HPC commands |
| Anything else | LLM answer using full conversation context |

## Quick start

```bash
git clone https://github.com/ArnavAK74/PEAT.git
cd PEAT
pip install -r requirements.txt
```

Create a `.env` file:

```env
# LLM (defaults to Jetstream2 Llama 4 Scout — no key needed)
LLM_BASE_URL=https://llm.jetstream-cloud.org/llama-4-scout/v1/
LLM_MODEL=llama-4-scout

# For harder reasoning tasks, use DeepSeek R1:
# LLM_BASE_URL=https://llm.jetstream-cloud.org/sglang/v1/
# LLM_MODEL=DeepSeek-R1

# RAG
UNPAYWALL_EMAIL=your@email.edu
SCIHUB_ENABLED=false

# HPC — Anvil (NAIRR allocation)
HPC_HOST=anvil.rcac.purdue.edu
HPC_USER=yournetid
HPC_PRIVATE_KEY=<raw private key content>
HPC_WORKDIR=/anvil/scratch/yournetid/peat_runs
HPC_PARTITION=gpu
HPC_QOS=gpu
HPC_ACCOUNT=your_allocation
```

```bash
streamlit run app.py
```

## LLM backends

| Endpoint | Model | Use case |
|----------|-------|----------|
| `https://llm.jetstream-cloud.org/llama-4-scout/v1/` | `llama-4-scout` | Default — general Q&A, annotation summaries |
| `https://llm.jetstream-cloud.org/sglang/v1/` | `DeepSeek-R1` | Harder reasoning tasks |

Both are Jetstream2-hosted, OpenAI-compatible, and require no API key.

## RAG pipeline

Paper retrieval is attempted in this order:
1. **Unpaywall** — open-access PDF (always on)
2. **Library cookie** — set `LIBRARY_COOKIE` env var for institutional access
3. **Sci-Hub** — set `SCIHUB_ENABLED=true` (dev only)

## HPC

Allowed command prefixes: `gmx`, `python`, `python3`, `bash`, `sh`, `squeue`, `sacct`, `sbatch`, `scancel`, `echo`, `ls`, `cat`, `head`, `tail`

Timeout: 300 s (override with `HPC_TIMEOUT_SECONDS`). Set `HPC_HOST` to run remotely; omit it to run commands locally.

## License

MIT
