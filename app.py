# app.py
import os
import re
import json
import requests

import streamlit as st
from openai import OpenAI

from data_fetch import (
    get_pdb_data,
    get_uniprot_ids_from_sifts,
    get_unpaywall_data,
    fetch_pdf_text,
    get_m_csa_active_sites,
    fetch_uniprot_features,
    get_pdb_id_from_sequence,
)
from structure_tools import build_3dmol_html
from sequence_tools import conservation_scores, run_blast
from predictors import predict_ddg_dynamut
from ui import plot_domains, plot_conservation
from hpc_tools import run_hpc_command
from bio_tools import foldseek_search, alphafold_fetch

# ── LLM client ─────────────────────────────────────────────────────────────────
_api_key  = os.getenv("OPENROUTER_API_KEY", "")
_base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
_model    = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
client    = OpenAI(api_key=_api_key, base_url=_base_url)
EMAIL     = os.getenv("UNPAYWALL_EMAIL", "")

_SYSTEM_PROMPT = (
    "You are PEAT, a protein engineering assistant for the KhareLab at Rutgers University. "
    "You help researchers analyze protein structures, interpret UniProt annotations, "
    "understand literature, and run GROMACS simulations on Amarel HPC. "
    "When referring to previously analyzed proteins, use the information from the conversation history."
)

_HPC_PREFIXES = {
    "gmx", "python", "python3", "bash", "sh",
    "squeue", "sacct", "sbatch", "scancel",
    "echo", "ls", "cat", "head", "tail",
}
# Matches messages whose intent is to fetch an AlphaFold structure:
#   "alphafold Q9WYE2" | "fetch alphafold Q9WYE2" | "af2 Q9WYE2"
_ALPHAFOLD_RE = re.compile(
    r'^\s*'
    r'(?:(?:fetch|get|show|load)\s+)?'
    r'(?:alphafold|af2|af)\s+'
    r'(?:for\s+)?'
    r'([A-Za-z0-9]+)'
    r'\s*$',
    re.IGNORECASE,
)

# Matches messages whose intent is to run a Foldseek search:
#   "foldseek 6B5X" | "similar structures 6B5X" | "find similar to 6B5X"
_FOLDSEEK_RE = re.compile(
    r'(?:^|.*\b)foldseek\b.*?(?:on|for|against|search)?\s+(?:pdb\s+)?([0-9][A-Za-z0-9]{3})\b'
    r'|(?:find\s+similar(?:\s+structures?)?(?:\s+to)?'
    r'|similar\s+structures?(?:\s+to)?'
    r'|structure\s+search(?:\s+for)?)\s+(?:pdb\s+)?([0-9][A-Za-z0-9]{3})\b',
    re.IGNORECASE,
)

# Matches messages whose *intent* is to analyze a PDB ID:
#   "6B5X" | "analyze 6B5X" | "fetch pdb 6B5X" | "look up 6B5X"
# Does NOT match PDB IDs embedded in longer questions.
_ANALYZE_RE = re.compile(
    r'^\s*'
    r'(?:(?:analyze|analyse|fetch|show|load|look\s*up|get|examine|study|open|run)\s+)?'
    r'(?:pdb\s+)?'
    r'([0-9][A-Za-z0-9]{3})'
    r'\s*$',
    re.IGNORECASE,
)

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PEAT – Protein Engineering Agent Toolkit")
st.title("🔬 PEAT – Protein Engineering Agent Toolkit")

# ── Session state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # LLM context — text only, passed on every call
    st.session_state.messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
if "chat_display" not in st.session_state:
    # Full display items including rich artifacts
    st.session_state.chat_display = []
if "analyzed_pdb_ids" not in st.session_state:
    # Set of PDB IDs already analyzed this session
    st.session_state.analyzed_pdb_ids = set()


# ── RAG helper ──────────────────────────────────────────────────────────────────
def _fetch_paper_text(doi: str, email: str) -> str | None:
    """Priority: Unpaywall → library cookie → Sci-Hub (dev only)."""
    if email:
        ua_data = get_unpaywall_data(doi, email)
        if ua_data:
            best    = ua_data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or ua_data.get("doi_url")
            if pdf_url:
                text = fetch_pdf_text(pdf_url)
                if text and len(text) > 200:
                    return text

    lib_cookie = os.getenv("LIBRARY_COOKIE", "")
    if lib_cookie:
        try:
            import fitz, tempfile
            r = requests.get(
                f"https://doi.org/{doi}",
                headers={"Cookie": lib_cookie},
                allow_redirects=True, timeout=15,
            )
            if r.ok and "pdf" in r.headers.get("Content-Type", "").lower():
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    tf.write(r.content)
                doc  = fitz.open(tf.name)
                text = "".join(p.get_text() for p in doc)[:10000]
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

    if os.getenv("SCIHUB_ENABLED", "false").lower() == "true":
        scihub_base = os.getenv("SCIHUB_URL", "https://sci-hub.se")
        try:
            r = requests.get(f"{scihub_base}/{doi}", timeout=20)
            if r.ok:
                match = re.search(r'src=["\']([^"\']*\.pdf[^"\']*)["\']', r.text)
                if match:
                    pdf_url = match.group(1)
                    if pdf_url.startswith("//"):
                        pdf_url = "https:" + pdf_url
                    text = fetch_pdf_text(pdf_url)
                    if text and len(text) > 200:
                        return text
        except Exception:
            pass

    return None


# ── Artifact renderer ───────────────────────────────────────────────────────────
def _render_artifact(artifact: dict) -> None:
    """Render a single artifact inside the current Streamlit container."""
    t = artifact["type"]
    if t == "html":
        st.components.v1.html(artifact["data"], height=550)
    elif t == "plotly":
        st.plotly_chart(artifact["data"], use_container_width=True)
    elif t == "code":
        st.code(artifact["data"], language=artifact.get("language", ""))
    elif t == "markdown":
        st.markdown(artifact["data"])
    elif t == "mutation_form":
        form_key = artifact.get("key", "mutate_form")
        with st.form(form_key):
            site     = st.text_input("Residue (e.g. A123)")
            mutation = st.text_input("Mutation (e.g. A123C)")
            submitted = st.form_submit_button("Predict ΔΔG")
            if submitted:
                try:
                    chain  = site[0]
                    resnum = int(site[1:])
                    result = predict_ddg_dynamut("temp.pdb", chain, resnum, mutation)
                    st.success(f"Predicted ΔΔG: {result.get('ddg')} kCal/mol")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif t == "expander":
        with st.expander(artifact["label"], expanded=artifact.get("expanded", False)):
            for sub in artifact["content"]:
                _render_artifact(sub)
    elif t == "tabs":
        tab_labels = [tab["label"] for tab in artifact["tabs"]]
        tab_objs   = st.tabs(tab_labels)
        for tab_obj, tab_def in zip(tab_objs, artifact["tabs"]):
            with tab_obj:
                for sub in tab_def["content"]:
                    _render_artifact(sub)


# ── Analysis pipeline ───────────────────────────────────────────────────────────
def _run_analysis(pdb_id: str, user_question: str) -> tuple[str, list]:
    """
    Fetch and process all data for a PDB ID.
    Returns (text_summary, artifacts) where text_summary goes into LLM context
    and artifacts are rendered in the chat message.
    """
    pdb_id = pdb_id.upper()

    entry = get_pdb_data(pdb_id)

    citation = entry.get("rcsb_primary_citation", {})
    doi      = citation.get("pdbx_database_id_doi", "N/A")
    title    = citation.get("title", "N/A")
    authors  = citation.get("rcsb_authors", [])
    journal  = citation.get("rcsb_journal_abbrev", "N/A")

    m_csa_sites = get_m_csa_active_sites(pdb_id)
    uniprot_ids = get_uniprot_ids_from_sifts(pdb_id)
    if uniprot_ids:
        uniprot_id  = uniprot_ids[0]
        up_features = fetch_uniprot_features(uniprot_id)
    else:
        uniprot_id  = None
        up_features = {"features": [], "comments": [], "proteinDescription": {}, "genes": []}

    # Try experimental structure; fall back to AlphaFold if unavailable
    af_result = None
    try:
        pdb_resp = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30
        )
        pdb_resp.raise_for_status()
        with open("temp.pdb", "wb") as f:
            f.write(pdb_resp.content)
        structure_source = "experimental"
    except Exception:
        if uniprot_id:
            af_result = alphafold_fetch(uniprot_id)
            import shutil
            shutil.copy(af_result["pdb_path"], "temp.pdb")
            structure_source = "alphafold"
        else:
            raise RuntimeError(
                f"Could not download experimental structure for {pdb_id} "
                "and no UniProt ID available for AlphaFold fallback."
            )

    # UniProt annotation summary via LLM
    gpt_summary = {}
    if up_features.get("comments"):
        all_texts = []
        for comment in up_features["comments"]:
            if comment.get("texts"):
                for text in comment["texts"]:
                    all_texts.append(text.get("value", ""))
            elif comment.get("commentType") == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {}).get("name", "")
                ec_num   = comment.get("reaction", {}).get("ecNumber", "")
                if reaction:
                    all_texts.append(f"Catalytic Activity: {reaction} (EC {ec_num})")
            elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    val = loc.get("location", {}).get("value", "")
                    if val:
                        all_texts.append(f"Subcellular Location: {val}")
            elif comment.get("commentType") == "INTERACTION":
                for interaction in comment.get("interactions", []):
                    g1    = interaction.get("interactantOne", {}).get("geneName", "")
                    g2    = interaction.get("interactantTwo", {}).get("geneName", "")
                    count = interaction.get("numberOfExperiments", 0)
                    all_texts.append(f"Interaction: {g1} ↔ {g2} ({count} experiments)")

        prompt = f"""You're an expert assistant for structural biologists.

Summarize the following UniProt annotations into three categories:
- Structure (domains, motifs, folding)
- Function (enzymatic activity, pathways)
- Sequence features (PTMs, polymorphisms, isoforms)

Return strictly as JSON with keys: "Structure", "Function", "Sequence". Each value is a list of bullet-point strings.

---
{chr(10).join(all_texts)}
---"""
        try:
            resp = client.chat.completions.create(
                model=_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            try:
                gpt_summary = json.loads(raw)
            except Exception:
                gpt_summary = {"Structure": [], "Function": [raw], "Sequence": []}
        except Exception:
            gpt_summary = {}

    # Literature Q&A
    lit_answer  = None
    paper_text  = None
    if doi != "N/A":
        paper_text = _fetch_paper_text(doi, EMAIL)
        if paper_text and user_question:
            qa_prompt = (
                f"You are an expert research assistant for protein engineers and biochemists.\n"
                f"Use the paper (DOI: {doi}) to answer the following question.\n\n"
                f"Question: {user_question}\n"
                f"---\nPaper Excerpt (first 10 000 chars):\n{paper_text}\n---\nAnswer:"
            )
            try:
                lit_answer = client.chat.completions.create(
                    model=_model,
                    messages=[{"role": "user", "content": qa_prompt}],
                    temperature=0.3,
                    max_tokens=1000,
                ).choices[0].message.content
            except Exception:
                lit_answer = None

    # Protein identity fields
    prot_desc = up_features.get("proteinDescription", {})
    rec_name  = prot_desc.get("recommendedName", {})
    name = rec_name.get("fullName", {}).get("value", "N/A")
    ec   = (rec_name.get("ecNumbers", [{}]) or [{}])[0].get("value", "N/A")
    gene = (
        up_features.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A")
        if up_features.get("genes") else "N/A"
    )

    # Text summary stored in LLM context
    text_summary = (
        f"Analysis complete for **{pdb_id}**.\n"
        f"- **Protein:** {name} | **Gene:** {gene} | **EC:** {ec}\n"
        f"- **Paper:** _{title}_ ({journal}) — DOI: {doi}\n"
    )
    if structure_source == "alphafold" and af_result:
        text_summary += (
            f"- **Structure:** AlphaFold model ({af_result['entry_id']}) — "
            f"no experimental PDB available. "
            f"Mean pLDDT: {af_result['plddt_mean']} "
            f"(min {af_result['plddt_min']}, max {af_result['plddt_max']})\n"
        )
    if gpt_summary.get("Function"):
        text_summary += "**Functional notes:**\n" + "\n".join(f"- {b}" for b in gpt_summary["Function"]) + "\n"
    if lit_answer:
        text_summary += f"**Literature answer:** {lit_answer[:400]}…\n"

    # ── Artifacts ──────────────────────────────────────────────────────────────
    # Tab 1: Literature & Catalysis
    tab1 = []
    tab1.append({"type": "markdown", "data": (
        "### Paper Metadata\n"
        f"- **DOI:** {doi}\n"
        f"- **Title:** {title}\n"
        f"- **Authors:** {', '.join(authors)}\n"
        f"- **Journal:** {journal}"
    )})
    if name != "N/A":
        tab1.append({"type": "markdown", "data": (
            "### UniProt Annotations\n"
            f"- **Protein:** {name}\n"
            f"- **EC Number:** {ec}\n"
            f"- **Gene:** {gene}"
        )})
    if gpt_summary.get("Function"):
        tab1.append({"type": "markdown", "data":
            "### Functional Roles\n" + "\n".join(f"- {b}" for b in gpt_summary["Function"])
        })
    if lit_answer:
        tab1.append({"type": "markdown", "data": f"### Literature Answer\n{lit_answer}"})
    elif doi != "N/A" and not paper_text:
        tab1.append({"type": "markdown", "data":
            "_No open-access PDF found (tried Unpaywall + library cookie + Sci-Hub)._"
        })

    # Tab 2: Sequence & Domains
    tab2 = []
    if uniprot_id:
        tab2.append({"type": "markdown", "data":
            f"**UniProt:** [{uniprot_id}](https://www.uniprot.org/uniprotkb/{uniprot_id})"
        })
    if up_features.get("features"):
        seq_len = entry.get("rcsb_entry_info", {}).get("polymer_monomer_count_maximum", 500)
        tab2.append({"type": "plotly", "data": plot_domains(up_features["features"], seq_len)})
    if gpt_summary.get("Sequence"):
        tab2.append({"type": "markdown", "data":
            "### Sequence Features\n" + "\n".join(f"- {b}" for b in gpt_summary["Sequence"])
        })
    if af_result:
        tab2.append({"type": "markdown", "data": (
            "### AlphaFold Confidence (pLDDT)\n"
            f"- **Entry:** {af_result['entry_id']}\n"
            f"- **Mean pLDDT:** {af_result['plddt_mean']} "
            f"(min {af_result['plddt_min']}, max {af_result['plddt_max']})\n"
            "_pLDDT > 90: very high confidence · 70–90: high · 50–70: low · < 50: very low_"
        )})
    if not tab2:
        tab2.append({"type": "markdown", "data": "_No UniProt domain annotations found._"})

    # Tab 3: Mutations & Predictions
    tab3 = [
        {"type": "markdown", "data": "### Mutation ΔΔG Predictions"},
        {"type": "mutation_form", "data": None, "key": f"mutate_{pdb_id}"},
    ]

    artifacts = [
        {"type": "tabs", "tabs": [
            {"label": "Literature & Catalysis", "content": tab1},
            {"label": "Sequence & Domains",      "content": tab2},
            {"label": "Mutations & Predictions", "content": tab3},
        ]},
        {"type": "markdown", "data": "### 3D Structure Viewer"},
        {"type": "html", "data": build_3dmol_html(pdb_id)},
    ]

    return text_summary, artifacts


# ── Intent helpers ──────────────────────────────────────────────────────────────
def _parse_analyze_request(text: str) -> str | None:
    """Return the PDB ID if the message is a request to analyze one, else None."""
    m = _ANALYZE_RE.match(text)
    return m.group(1).upper() if m else None

def _parse_foldseek_request(text: str) -> str | None:
    """Return the PDB ID if the message is a Foldseek search request, else None."""
    m = _FOLDSEEK_RE.search(text)
    if not m:
        return None
    return (m.group(1) or m.group(2)).upper()

def _parse_alphafold_request(text: str) -> str | None:
    """Return the UniProt ID if the message is an AlphaFold fetch request, else None."""
    m = _ALPHAFOLD_RE.match(text)
    return m.group(1).upper() if m else None

def _is_hpc_command(text: str) -> bool:
    parts = text.strip().split()
    return bool(parts) and parts[0].lower() in _HPC_PREFIXES

_AA_CHARS     = set("ACDEFGHIKLMNPQRSTVWY")
_AA_THRESHOLD = 0.85   # fraction of non-whitespace chars that must be valid AAs
_MIN_SEQ_LEN  = 20     # ignore anything shorter (avoids false positives on short words)

def _parse_sequence_input(text: str) -> str | None:
    """
    Return the cleaned amino acid sequence if the message is a FASTA block or a
    raw AA sequence; else None.

    Handles:
    - FASTA format: strips all header lines starting with '>'
    - Raw sequence: accepts if >= 85% of characters are standard amino acids
      and the sequence is at least 20 residues long
    """
    lines = text.strip().splitlines()
    seq_lines = [l for l in lines if not l.startswith(">")]
    seq = "".join(seq_lines).replace(" ", "").upper()

    if len(seq) < _MIN_SEQ_LEN:
        return None
    valid = sum(1 for c in seq if c in _AA_CHARS)
    if valid / len(seq) >= _AA_THRESHOLD:
        return seq
    return None


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search by Sequence")
    seq_input = st.text_area("Protein sequence (FASTA or plain AA)", height=140)
    if st.button("Find & Analyze PDB"):
        if seq_input.strip():
            with st.spinner("Searching RCSB for matching PDB…"):
                found_id = get_pdb_id_from_sequence(seq_input.strip())
            if found_id:
                st.success(f"Found: {found_id}")
                st.session_state._pending_pdb = found_id
                st.session_state._pending_question = "What is the function of this protein?"
            else:
                st.error("No matching PDB found.")
        else:
            st.warning("Enter a sequence first.")

    st.markdown("---")
    st.caption(
        "Or type a PDB ID (e.g. `Analyze 6B5X`) or an HPC command "
        "(e.g. `gmx mdrun -deffnm em`) directly in the chat below."
    )


# ── Render chat history ─────────────────────────────────────────────────────────
for item in st.session_state.chat_display:
    with st.chat_message(item["role"]):
        if item.get("content"):
            st.markdown(item["content"])
        for artifact in item.get("artifacts", []):
            _render_artifact(artifact)


# ── Handle sequence → PDB injection from sidebar ───────────────────────────────
if hasattr(st.session_state, "_pending_pdb"):
    pdb_id   = st.session_state.pop("_pending_pdb")
    question = st.session_state.pop("_pending_question", "")
    user_msg = f"Analyze {pdb_id}" + (f" — {question}" if question else "")

    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.chat_display.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        if pdb_id in st.session_state.analyzed_pdb_ids:
            with st.spinner("Thinking…"):
                try:
                    resp = client.chat.completions.create(
                        model=_model,
                        messages=st.session_state.messages,
                        temperature=0.3,
                        max_tokens=1000,
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"LLM error: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_display.append({"role": "assistant", "content": answer})
        else:
            with st.spinner(f"Analyzing {pdb_id}…"):
                try:
                    text_summary, artifacts = _run_analysis(pdb_id, question)
                    st.markdown(text_summary)
                    for artifact in artifacts:
                        _render_artifact(artifact)
                    st.session_state.analyzed_pdb_ids.add(pdb_id)
                    st.session_state.messages.append({"role": "assistant", "content": text_summary})
                    st.session_state.chat_display.append({
                        "role": "assistant", "content": text_summary, "artifacts": artifacts,
                    })
                except Exception as e:
                    err = f"Analysis failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.session_state.chat_display.append({"role": "assistant", "content": err})
    st.rerun()


# ── Chat input loop ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a protein (e.g. 'Analyze 6B5X') or run an HPC command…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pdb_id        = _parse_analyze_request(prompt)
    foldseek_id   = _parse_foldseek_request(prompt)
    alphafold_id  = _parse_alphafold_request(prompt)
    sequence      = _parse_sequence_input(prompt)

    # Route: HPC command (checked first — unambiguous prefix match)
    if _is_hpc_command(prompt):
        with st.chat_message("assistant"):
            with st.spinner("Running HPC command…"):
                hpc_output = run_hpc_command(prompt.strip())
            response = f"**HPC Output:**\n```bash\n{hpc_output}\n```"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": hpc_output})
            st.session_state.chat_display.append({"role": "assistant", "content": response})

    # Route: AlphaFold structure fetch
    elif alphafold_id:
        with st.chat_message("assistant"):
            with st.spinner(f"Fetching AlphaFold structure for {alphafold_id}…"):
                try:
                    af = alphafold_fetch(alphafold_id)
                    response = (
                        f"### AlphaFold Structure — {alphafold_id}\n"
                        f"- **Entry:** {af['entry_id']}\n"
                        f"- **Gene:** {af['gene']} | **Organism:** {af['organism']}\n"
                        f"- **Description:** {af['description']}\n"
                        f"- **Mean pLDDT:** {af['plddt_mean']} "
                        f"(min {af['plddt_min']}, max {af['plddt_max']})\n\n"
                        "_pLDDT > 90: very high · 70–90: high · 50–70: low · < 50: very low_"
                    )
                    viewer_html = build_3dmol_html(alphafold_id, pdb_path=af["pdb_path"])
                    artifacts = [
                        {"type": "markdown", "data": response},
                        {"type": "markdown", "data": "### 3D Structure Viewer"},
                        {"type": "html",     "data": viewer_html},
                    ]
                    st.markdown(response)
                    _render_artifact(artifacts[1])
                    _render_artifact(artifacts[2])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_display.append({
                        "role": "assistant", "content": "", "artifacts": artifacts,
                    })
                except Exception as e:
                    err = f"AlphaFold fetch failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.session_state.chat_display.append({"role": "assistant", "content": err})

    # Route: Foldseek structural similarity search
    elif foldseek_id:
        with st.chat_message("assistant"):
            with st.spinner(f"Running Foldseek search for {foldseek_id}…"):
                try:
                    hits = foldseek_search(foldseek_id)
                    if hits:
                        rows = ["| # | PDB/AF ID | TM-score | E-value | Seq. Identity | Description |",
                                "|---|-----------|----------|---------|---------------|-------------|"]
                        for i, h in enumerate(hits, 1):
                            tm   = f"{h['tm_score']:.3f}"        if h["tm_score"]            is not None else "—"
                            ev   = f"{h['evalue']:.2e}"          if h["evalue"]              is not None else "—"
                            sid  = f"{h['sequence_identity']:.1%}" if h["sequence_identity"] is not None else "—"
                            rows.append(f"| {i} | `{h['pdb_id']}` | {tm} | {ev} | {sid} | {h['description']} |")
                        table_md = "\n".join(rows)
                        response = f"### Foldseek — Top hits for {foldseek_id}\n\n{table_md}"
                    else:
                        response = f"Foldseek returned no hits for {foldseek_id}."

                    artifact = {"type": "markdown", "data": response}
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_display.append({
                        "role": "assistant", "content": "", "artifacts": [artifact],
                    })
                except Exception as e:
                    err = f"Foldseek search failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.session_state.chat_display.append({"role": "assistant", "content": err})

    # Route: new protein analysis
    elif pdb_id and pdb_id not in st.session_state.analyzed_pdb_ids:
        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing {pdb_id}…"):
                try:
                    text_summary, artifacts = _run_analysis(pdb_id, prompt)
                    st.markdown(text_summary)
                    for artifact in artifacts:
                        _render_artifact(artifact)
                    st.session_state.analyzed_pdb_ids.add(pdb_id)
                    st.session_state.messages.append({"role": "assistant", "content": text_summary})
                    st.session_state.chat_display.append({
                        "role": "assistant", "content": text_summary, "artifacts": artifacts,
                    })
                except Exception as e:
                    err = f"Analysis failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.session_state.chat_display.append({"role": "assistant", "content": err})

    # Route: sequence input (FASTA or raw AA) → BLAST → analysis
    elif sequence and not pdb_id:
        with st.chat_message("assistant"):
            with st.spinner("Searching RCSB for a matching PDB…"):
                found_id = get_pdb_id_from_sequence(sequence)
            if not found_id:
                err = "No matching PDB found for the provided sequence."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.session_state.chat_display.append({"role": "assistant", "content": err})
            elif found_id in st.session_state.analyzed_pdb_ids:
                # Already analyzed — let the LLM answer from context
                with st.spinner("Thinking…"):
                    try:
                        resp = client.chat.completions.create(
                            model=_model,
                            messages=st.session_state.messages,
                            temperature=0.3,
                            max_tokens=1000,
                        )
                        answer = resp.choices[0].message.content
                    except Exception as e:
                        answer = f"LLM error: {e}"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_display.append({"role": "assistant", "content": answer})
            else:
                with st.spinner(f"Analyzing {found_id}…"):
                    try:
                        text_summary, artifacts = _run_analysis(found_id, prompt)
                        st.markdown(text_summary)
                        for artifact in artifacts:
                            _render_artifact(artifact)
                        st.session_state.analyzed_pdb_ids.add(found_id)
                        st.session_state.messages.append({"role": "assistant", "content": text_summary})
                        st.session_state.chat_display.append({
                            "role": "assistant", "content": text_summary, "artifacts": artifacts,
                        })
                    except Exception as e:
                        err = f"Analysis failed: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        st.session_state.chat_display.append({"role": "assistant", "content": err})

    # Route: LLM with full history (questions, follow-ups, already-analyzed PDB IDs)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = client.chat.completions.create(
                        model=_model,
                        messages=st.session_state.messages,
                        temperature=0.3,
                        max_tokens=1000,
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"LLM error: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_display.append({"role": "assistant", "content": answer})
