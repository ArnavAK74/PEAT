# app.py
import os
import re
import requests
import json

import streamlit as st
from openai import OpenAI  # OpenRouter is OpenAI-compatible

from data_fetch       import (
    get_pdb_data,
    get_uniprot_ids_from_sifts,
    get_unpaywall_data,
    fetch_pdf_text,
    get_m_csa_active_sites,
    fetch_uniprot_features,
    get_pdb_id_from_sequence
)
from structure_tools  import build_3dmol_html, find_hotspots
from sequence_tools   import conservation_scores, run_blast
from predictors       import predict_ddg_dynamut
from ui               import plot_domains, plot_conservation, show_mutation_form
from hpc_tools        import run_hpc_command

# ── LLM client ────────────────────────────────────────────────────────────────
# Dev:  set OPENROUTER_API_KEY + LLM_BASE_URL=https://openrouter.ai/api/v1
# Prod: set LLM_BASE_URL to your HPC-hosted endpoint (e.g. vLLM/Ollama on HPC)
_api_key  = os.getenv("OPENROUTER_API_KEY", "")
_base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
_model    = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

client = OpenAI(api_key=_api_key, base_url=_base_url)
EMAIL  = os.getenv("UNPAYWALL_EMAIL", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PEAT – Protein Engineering Agent Toolkit")
st.title("🔬 PEAT – Protein Engineering Agent Toolkit")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Input")
    input_type = st.radio("Select Input Type:", ["PDB ID", "Protein Sequence"])
    if input_type == "PDB ID":
        pdb_id   = st.text_input("Enter PDB ID (e.g., 6B5X)").strip()
        sequence = None
    else:
        sequence = st.text_area("Enter Protein Sequence (FASTA or plain AA)").strip()
        pdb_id   = None

    if sequence and not pdb_id:
        with st.spinner("🔍 Searching for matching PDB ID..."):
            pdb_id = get_pdb_id_from_sequence(sequence)
            if pdb_id:
                st.success(f"✅ Found matching PDB ID: {pdb_id}")
            else:
                st.error("❌ Could not find a matching PDB ID for the sequence.")
                st.stop()

    user_question = st.text_area("Your question:", "What is the function of the protein?")

    st.markdown("---")
    st.header("⚙️ HPC Command")
    hpc_cmd = st.text_input(
        "GROMACS / HPC command",
        placeholder="gmx mdrun -deffnm minimize",
        help="Short jobs only (<5 min / 16 CPU). Runs on the connected HPC resource."
    )
    run_hpc = st.button("▶ Run HPC Command")
    run     = st.button("🔎 Analyze Protein")


# ── HPC block ─────────────────────────────────────────────────────────────────
if run_hpc and hpc_cmd.strip():
    with st.spinner("⚙️ Submitting HPC job..."):
        hpc_result = run_hpc_command(hpc_cmd.strip())
    st.subheader("HPC Output")
    st.code(hpc_result, language="bash")


# ── RAG helper ────────────────────────────────────────────────────────────────
def _fetch_paper_text(doi: str, email: str) -> str | None:
    """
    Priority order:
    1. Unpaywall open-access PDF (production-safe)
    2. Library auth cookie via LIBRARY_COOKIE env var (production)
    3. Sci-Hub (dev only — set SCIHUB_ENABLED=true to activate)
    """
    # 1. Unpaywall
    if email:
        ua_data = get_unpaywall_data(doi, email)
        if ua_data:
            best    = ua_data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or ua_data.get("doi_url")
            if pdf_url:
                text = fetch_pdf_text(pdf_url)
                if text and len(text) > 200:
                    return text

    # 2. Library auth cookie (cf. Zotero connector approach)
    lib_cookie = os.getenv("LIBRARY_COOKIE", "")
    if lib_cookie:
        try:
            import fitz, tempfile
            r = requests.get(f"https://doi.org/{doi}",
                             headers={"Cookie": lib_cookie},
                             allow_redirects=True, timeout=15)
            if r.ok and "pdf" in r.headers.get("Content-Type", "").lower():
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    tf.write(r.content)
                doc  = fitz.open(tf.name)
                text = "".join(p.get_text() for p in doc)[:10000]
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

    # 3. Sci-Hub fallback (dev only)
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


# ── Main analysis ─────────────────────────────────────────────────────────────
if run:
    try:
        entry = get_pdb_data(pdb_id)

        pdb_url  = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        pdb_resp = requests.get(pdb_url)
        pdb_resp.raise_for_status()
        with open("temp.pdb", "wb") as pdb_file:
            pdb_file.write(pdb_resp.content)

        doi     = entry["rcsb_primary_citation"].get("pdbx_database_id_doi", "N/A")
        title   = entry["rcsb_primary_citation"].get("title", "N/A")
        authors = entry["rcsb_primary_citation"].get("rcsb_authors", [])
        journal = entry["rcsb_primary_citation"].get("rcsb_journal_abbrev", "N/A")

        m_csa_sites = get_m_csa_active_sites(pdb_id)

        uniprot_ids = get_uniprot_ids_from_sifts(pdb_id)
        if uniprot_ids:
            uniprot_id  = uniprot_ids[0]
            up_features = fetch_uniprot_features(uniprot_id)
        else:
            uniprot_id  = None
            up_features = {"features": [], "comments": [], "proteinDescription": {}, "genes": []}

        hotspots    = find_hotspots("temp.pdb")
        gpt_summary = {}

        # ── UniProt annotation summary via LLM ────────────────────────────────
        if up_features.get("comments"):
            all_texts = []
            for comment in up_features["comments"]:
                if comment.get("texts"):
                    for text in comment["texts"]:
                        all_texts.append(text.get("value", ""))
                elif comment.get("commentType") == "CATALYTIC ACTIVITY":
                    reaction = comment.get("reaction", {}).get("name", "")
                    ec       = comment.get("reaction", {}).get("ecNumber", "")
                    if reaction:
                        all_texts.append(f"Catalytic Activity: {reaction} (EC {ec})")
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

            combined_text = "\n".join(all_texts)
            prompt = f"""
You're an expert assistant for structural biologists.

Summarize the following UniProt annotations into three categories:
- Structure (domains, motifs, folding)
- Function (enzymatic activity, pathways)
- Sequence features (PTMs, polymorphisms, isoforms)

Return strictly as JSON with keys: "Structure", "Function", "Sequence". Each value is a list of bullet-point strings.

---
{combined_text}
---
"""
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
            except Exception as e:
                st.warning(f"LLM annotation summary failed: {e}")
        # ── Layout ────────────────────────────────────────────────────────────
        st.subheader(f"Results for {pdb_id.upper()}")
        tab1, tab2, tab3 = st.tabs(["Literature & Catalysis", "Sequence & Domains", "Mutations & Predictions"])

        with tab1:
            st.markdown("### 📄 Paper Metadata")
            st.markdown(f"- **DOI:** {doi}")
            st.markdown(f"- **Title:** {title}")
            st.markdown(f"- **Authors:** {', '.join(authors)}")
            st.markdown(f"- **Journal:** {journal}")

            if up_features and "proteinDescription" in up_features:
                st.markdown("### 🧬 UniProt Functional Annotations")
                name = (up_features.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {}).get("value", "N/A"))
                ec   = (up_features.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("ecNumbers", [{}])[0].get("value", "N/A"))
                genes_list = up_features.get("genes", [])
                gene       = genes_list[0].get("geneName", {}).get("value", "N/A") if genes_list else "N/A"
                st.markdown(f"- **Protein Name:** {name}")
                st.markdown(f"- **EC Number:** {ec}")
                st.markdown(f"- **Gene:** {gene}")

            with st.expander("🧬 Functional Roles (Click to expand)"):
                for func in gpt_summary.get("Function", []):
                    st.markdown(f"- {func}")

            st.markdown("### 🧠 LLM Answer to Your Question")

            if doi != "N/A":
                paper_text = _fetch_paper_text(doi, EMAIL)
                if paper_text:
                    prompt = f"""
You are an expert research assistant for protein engineers and biochemists.
Use the paper (DOI: {doi}) to answer the following question.

Question: {user_question}
---
Paper Excerpt (first 10 000 chars):
{paper_text}
---
Answer:"""
                    try:
                        answer = client.chat.completions.create(
                            model=_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=1000,
                        ).choices[0].message.content

                        st.success("Answer generated:")
                        st.write(answer)

                        with st.expander("📄 Show Paper Excerpt (first 10 000 chars)"):
                            st.write(paper_text)
                    except Exception as e:
                        st.warning(f"LLM failed: {e}")
                else:
                    st.warning("No open-access PDF found (tried Unpaywall + library cookie + Sci-Hub).")
            else:
                st.warning("DOI not found; skipping LLM literature summary.")

        with tab2:
            st.markdown("### Sequence Features & Domains")
            if uniprot_id:
                st.markdown(f"**UniProt Accession**: [{uniprot_id}](https://www.uniprot.org/uniprotkb/{uniprot_id})")
            if up_features and up_features.get("features"):
                st.plotly_chart(
                    plot_domains(up_features["features"],
                                 entry["rcsb_entry_info"]["polymer_monomer_count_maximum"]),
                    use_container_width=True
                )
                with st.expander("🧬 Sequence Information (Click to expand)"):
                    for seq_item in gpt_summary.get("Sequence", []):
                        st.markdown(f"- {seq_item}")
            else:
                st.write("🔍 No UniProt domain annotations found.")

        with tab3:
            st.markdown("### Structural Hotspots")
            st.write(hotspots or "No hotspots detected.")
            st.markdown("---")
            st.markdown("### Mutation ΔΔG Predictions")
            show_mutation_form()

        st.markdown("### 🧬 3D Structure Viewer")
        st.components.v1.html(build_3dmol_html(pdb_id), height=550)

    except Exception as e:
        st.error(f"❌ Error: {e}")
