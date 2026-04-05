# bio_tools.py
import os
import time
import requests

_FOLDSEEK_URL = "https://search.foldseek.com/api"
_CACHE_DIR    = "/tmp/peat_structures"


def foldseek_search(pdb_id: str) -> dict:
    """
    Search for structurally similar proteins using the Foldseek REST API.

    Steps:
      1. Cache PDB file in /tmp/peat_structures/, downloading from RCSB if absent.
      2. Submit to Foldseek against pdb100 + afdb50 databases.
      3. Poll the ticket endpoint until results are ready.
      4. Return a dict with:
           hits  — top 10 hits sorted by TM-score (descending), each with keys:
                   pdb_id, evalue, tm_score, sequence_identity, description
           raw   — full API response payload for debugging
    """
    pdb_id = pdb_id.upper()
    os.makedirs(_CACHE_DIR, exist_ok=True)

    # ── 1. Cache / download PDB file ────────────────────────────────────────────
    pdb_path = os.path.join(_CACHE_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        r = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            timeout=30,
        )
        r.raise_for_status()
        with open(pdb_path, "wb") as fh:
            fh.write(r.content)

    # ── 2. Submit ticket ─────────────────────────────────────────────────────────
    with open(pdb_path, "rb") as fh:
        resp = requests.post(
            f"{_FOLDSEEK_URL}/ticket",
            data={"database[]": ["pdb100", "afdb50"], "mode": "3diaa"},
            files={"q": (f"{pdb_id}.pdb", fh, "chemical/x-pdb")},
            timeout=30,
        )
    resp.raise_for_status()
    ticket_id = resp.json()["id"]

    # ── 3. Poll until complete (max 5 min, 5 s intervals) ───────────────────────
    for _ in range(60):
        time.sleep(5)
        poll = requests.get(f"{_FOLDSEEK_URL}/ticket/{ticket_id}", timeout=15)
        poll.raise_for_status()
        status_data = poll.json()
        status = status_data.get("status", "")
        if status == "COMPLETE":
            break
        if status == "ERROR":
            raise RuntimeError(f"Foldseek job failed for {pdb_id}: {status_data}")
    else:
        raise TimeoutError(f"Foldseek job timed out after 5 minutes for {pdb_id}")

    # ── 4. Fetch results from the result endpoint ────────────────────────────────
    result_resp = requests.get(f"{_FOLDSEEK_URL}/result/{ticket_id}/0", timeout=30)
    result_resp.raise_for_status()
    data = result_resp.json()

    # ── 5. Parse hits ────────────────────────────────────────────────────────────
    hits = []
    for db_result in data.get("results", []):
        # alignments is a list-of-lists: one inner list per query sequence
        for query_alignments in db_result.get("alignments", []):
            if not isinstance(query_alignments, list):
                query_alignments = [query_alignments]
            for hit in query_alignments:
                hits.append({
                    "pdb_id":            hit.get("target", ""),
                    "evalue":            hit.get("eval"),
                    "tm_score":          hit.get("score"),
                    "sequence_identity": (hit["seqId"] / 100) if hit.get("seqId") is not None else None,
                    "description":       hit.get("taxName") or hit.get("description", ""),
                })

    hits.sort(key=lambda h: h["tm_score"] or 0.0, reverse=True)
    return {"hits": hits[:10], "raw": data}
