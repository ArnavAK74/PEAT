# PEAT – Protein Engineering Agent Toolkit

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What PEAT is

PEAT is a RAG-powered structural biology assistant for the KhareLab at Rutgers University, built as a Streamlit web app. It combines structure analysis (PDB/UniProt), literature retrieval with LLM Q&A, and HPC job execution in a single chatbot interface.

- **Repo**: https://github.com/ArnavAK74/PEAT.git
- **Landing page**: https://arnavak74.github.io/PEAT

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file (or `.streamlit/secrets.toml`) with:

```env
# LLM
OPENROUTER_API_KEY=your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/auto

# RAG
UNPAYWALL_EMAIL=your@email.edu
SCIHUB_ENABLED=false        # true for dev fallback only
LIBRARY_COOKIE=             # institutional session cookie for prod

# HPC (optional)
HPC_HOST=amarel.rutgers.edu
HPC_USER=yournetid
HPC_PRIVATE_KEY=            # raw private key content (not a file path)
HPC_WORKDIR=/scratch/<netid>/peat_runs
```

Note: `HPC_PRIVATE_KEY` holds the raw key content (not a path). `hpc_tools.py` writes it to a `chmod 600` temp file and deletes it after each command.

No build step, no test suite, no linter configured.

## File structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit entrypoint — chatbot interface, routing, session state |
| `ui.py` | Plotly domain map (`plot_domains`), conservation chart, mutation form |
| `data_fetch.py` | RCSB PDB, UniProt/SIFTS, Unpaywall, Sci-Hub, M-CSA fetches |
| `structure_tools.py` | `build_3dmol_html` — 3Dmol.js viewer HTML only |
| `sequence_tools.py` | NCBI BLAST, MSA conservation scoring |
| `hpc_tools.py` | SSH-based HPC command execution with whitelist and timeout |
| `predictors.py` | ΔΔG and binding-change prediction API calls (placeholder endpoints) |
| `bio_tools.py` | Foldseek, AlphaFold DB, STRING, InterPro — all REST, no compute |

## Architecture

### Chatbot interface (`app.py`)

The app is fully chatbot-driven. There is no Analyze button.

**Session state:**
- `st.session_state.messages` — text-only LLM context, prefixed with a system prompt, passed in full on every LLM call
- `st.session_state.chat_display` — parallel list of display items, each `{"role", "content", "artifacts": [...]}`. Re-rendered on every Streamlit rerun.
- `st.session_state.analyzed_pdb_ids` — set of PDB IDs already analyzed this session

**Message routing (in order):**
1. HPC prefix match (`gmx`, `sbatch`, `squeue`, etc.) → `run_hpc_command()`
2. `_parse_analyze_request()` matches **and** PDB ID not yet in `analyzed_pdb_ids` → `_run_analysis()` pipeline
3. Everything else → LLM with full `messages` history (covers follow-up questions and re-mentions of already-analyzed proteins)

**Analyze intent detection:** `_ANALYZE_RE` is anchored (`^...$`) and only matches when the entire message is a PDB ID optionally preceded by an action verb (`analyze`, `fetch`, `show`, `look up`, etc.). PDB IDs mentioned inside longer questions do not trigger re-analysis.

**Rich artifacts** are stored as plain dicts in `chat_display` and rendered by `_render_artifact()`. Types: `tabs`, `plotly`, `html`, `markdown`, `code`, `mutation_form`. Plotly figures are stored as live Python objects in session state.

**Mutation form** uses a per-PDB key (`mutate_{pdb_id}`) to avoid duplicate `st.form` key errors across multiple analyses in one session.

### Analysis pipeline (`_run_analysis`)

Called once per PDB ID per session. Steps:
1. Fetch PDB entry JSON from RCSB → download PDB file to `temp.pdb`
2. Fetch UniProt IDs via SIFTS → fetch UniProt features and comments
3. Build UniProt annotation summary via LLM (JSON with Structure / Function / Sequence keys; strips markdown fences before `json.loads`, falls back gracefully)
4. Fetch paper text via RAG pipeline (Unpaywall → library cookie → Sci-Hub)
5. Run literature Q&A if paper text found
6. Return `(text_summary, artifacts)` — summary goes into LLM context, artifacts render in chat

### LLM routing

| Environment | `LLM_BASE_URL` | Model | Notes |
|-------------|---------------|-------|-------|
| Prod (default) | `https://llm.jetstream-cloud.org/llama-4-scout/v1/` | `llama-4-scout` | Jetstream2-hosted Llama 4 Scout |
| Prod (reasoning) | `https://llm.jetstream-cloud.org/sglang/v1/` | `DeepSeek-R1` | Jetstream2-hosted DeepSeek R1 — use for harder reasoning tasks |
| Dev | `https://openrouter.ai/api/v1` | any | Free-tier models via OpenRouter |

Both Jetstream2 endpoints are OpenAI-compatible and no API key is required (set `OPENROUTER_API_KEY=""` or leave unset). Client is OpenAI-compatible (`openai` SDK).

### RAG pipeline order

1. **Unpaywall** — open-access PDF (always on)
2. **Library auth cookie** — set `LIBRARY_COOKIE` env var
3. **Sci-Hub** — set `SCIHUB_ENABLED=true` (dev only)

### HPC (`hpc_tools.py`)

Allowed command prefixes: `gmx`, `python`, `python3`, `bash`, `sh`, `squeue`, `sacct`, `sbatch`, `scancel`, `echo`, `ls`, `cat`, `head`, `tail`

Timeout: `HPC_TIMEOUT_SECONDS` (default 300 s). When `HPC_HOST` is unset, commands run locally.

SSH options always include `-o StrictHostKeyChecking=no -o BatchMode=yes` to prevent interactive prompts on Streamlit Cloud.

## What still needs to be built

### Tool-calling loop
The LLM should be able to invoke tools by name:
- `fetch_structure(pdb_id)` — PDB + UniProt lookup
- `fetch_paper(doi)` — RAG pipeline
- `run_hpc(command)` — submit job to Amarel
- `monitor_hpc(job_id)` — squeue check
- `download_hpc(file_path)` — pull output file from Amarel
- `predict_mutation(pdb, chain, resnum, mutation)` — ΔΔG
- `foldseek_search(pdb_id)` — structure similarity (REST, `bio_tools.py`)
- `alphafold_fetch(uniprot_id)` — precomputed structure from AlphaFold DB
- `string_interactions(uniprot_id)` — protein interaction network
- `interpro_domains(uniprot_id)` — domain/family annotations

### Multimodal context chain
RAG(input) → fold(input) → database_call(fold) → RAG(database_results) → synthesized LLM response

## Bio tools API notes (all free, no auth)

**Foldseek** — `POST https://search.foldseek.com/api/ticket`, databases: `pdb100`, `afdb50`, `swissprot`, returns TM-score + e-value

**AlphaFold DB** — `GET https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}`, returns CIF/PDB URL + pLDDT scores

**STRING** — `GET https://string-db.org/api/json/network`, params: `identifiers`, `species=9606`, `required_score=400`

**InterPro** — `GET https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_id}`
