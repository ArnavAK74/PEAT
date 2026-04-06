# hpc_tools.py
"""
HPC command execution tool for PEAT.

Dev mode  : runs the command locally (useful for testing gmx on a workstation).
Prod mode : SSH into the HPC login node and submit via the SLURM/PBS wrapper
            configured in HPC_HOST / HPC_USER / HPC_PRIVATE_KEY env vars.

Safety guardrails
-----------------
- Only whitelisted prefixes are allowed (gmx, python, bash, squeue, sacct …).
- Timeout is capped at HPC_TIMEOUT_SECONDS (default 300 s = 5 min).
- Output is truncated to 4 000 chars to fit in the Streamlit sidebar.
"""

import os
import re
import stat
import subprocess
import shlex
import tempfile
from contextlib import contextmanager

# ── Config ────────────────────────────────────────────────────────────────────
_ALLOWED_PREFIXES = {
    "gmx", "python", "python3", "bash", "sh",
    "squeue", "sacct", "sbatch", "scancel",
    "echo", "ls", "cat", "head", "tail",
}
_TIMEOUT          = int(os.getenv("HPC_TIMEOUT_SECONDS", "300"))
_MAX_OUTPUT_CHARS = 4000
_CACHE_DIR        = "/tmp/peat_structures"

# SSH / SCP settings
_HPC_HOST       = os.getenv("HPC_HOST", "")
_HPC_USER       = os.getenv("HPC_USER", "")
_HPC_WORKDIR    = os.getenv("HPC_WORKDIR", "~/peat_runs")
_HPC_PARTITION  = os.getenv("HPC_PARTITION", "shared")        # Anvil default
_GROMACS_MODULE = os.getenv("GROMACS_MODULE", "gromacs/2024.1")


# ── Key-file context manager ─────────────────────────────────────────────────
@contextmanager
def _key_file():
    """
    Write HPC_PRIVATE_KEY to a chmod-600 temp file, yield its path,
    then delete it — even if an exception is raised.
    """
    private_key = os.getenv("HPC_PRIVATE_KEY", "")
    if not private_key:
        raise RuntimeError("HPC_PRIVATE_KEY env var is not set.")
    private_key = private_key.replace("\\n", "\n")
    if not private_key.endswith("\n"):
        private_key += "\n"

    print(f"[HPC] key first 20: {repr(private_key[:20])}")
    print(f"[HPC] key last  20: {repr(private_key[-20:])}")

    fd, path = tempfile.mkstemp(prefix="peat_hpc_", suffix=".pem")
    try:
        os.write(fd, private_key.encode())
        os.close(fd)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Low-level helpers ─────────────────────────────────────────────────────────
def _run_local(cmd: str, timeout: int = _TIMEOUT) -> str:
    """Run the command in a local subprocess, return stdout+stderr truncated."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = f"ERROR: command timed out after {timeout} s."
    except Exception as e:
        output = f"ERROR: {e}"
    return output[:_MAX_OUTPUT_CHARS]


def _ssh(remote_cmd: str, workdir: str | None = None) -> str:
    """
    Run a single command on HPC_HOST via SSH.
    If workdir is given, cd into it first.
    """
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured.")
    prefix = f"cd {workdir} && " if workdir else ""
    with _key_file() as kp:
        ssh_cmd = (
            f"ssh -i {kp} "
            f"-o StrictHostKeyChecking=no -o BatchMode=yes "
            f"{_HPC_USER}@{_HPC_HOST} "
            f"'{prefix}{remote_cmd}'"
        )
        return _run_local(ssh_cmd)


def _scp_to(local_path: str, remote_path: str) -> str:
    """SCP a local file to HPC_HOST:remote_path."""
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured.")
    with _key_file() as kp:
        cmd = (
            f"scp -i {kp} "
            f"-o StrictHostKeyChecking=no -o BatchMode=yes "
            f"{local_path} {_HPC_USER}@{_HPC_HOST}:{remote_path}"
        )
        return _run_local(cmd, timeout=120)


def _scp_from(remote_glob: str, local_dir: str) -> str:
    """SCP files matching remote_glob from HPC_HOST into local_dir."""
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured.")
    os.makedirs(local_dir, exist_ok=True)
    with _key_file() as kp:
        cmd = (
            f"scp -i {kp} "
            f"-o StrictHostKeyChecking=no -o BatchMode=yes "
            f'"{_HPC_USER}@{_HPC_HOST}:{remote_glob}" '
            f"{local_dir}/"
        )
        return _run_local(cmd, timeout=120)


# ── General HPC command (existing public API) ─────────────────────────────────
def run_hpc_command(cmd: str) -> str:
    """
    Execute a short HPC command and return stdout+stderr as a string.

    Routing:
    - If HPC_HOST is set → SSH to the HPC login node.
    - Otherwise         → run locally (dev / workstation with gmx installed).
    """
    try:
        first_token = shlex.split(cmd)[0].split("/")[-1]
    except ValueError:
        return "ERROR: could not parse command."

    if first_token not in _ALLOWED_PREFIXES:
        return (
            f"ERROR: command '{first_token}' is not in the allowed list.\n"
            f"Allowed: {', '.join(sorted(_ALLOWED_PREFIXES))}"
        )

    if _HPC_HOST:
        return _run_remote(cmd)
    else:
        return _run_local(cmd)


def _run_remote(cmd: str) -> str:
    """SSH into HPC_HOST and run the command in HPC_WORKDIR."""
    try:
        return _ssh(cmd, workdir=_HPC_WORKDIR)
    except RuntimeError as e:
        return f"ERROR: {e}"


# ── GROMACS energy minimization workflow ──────────────────────────────────────

_EM_MDP = """\
; Steepest-descent energy minimization — vacuum, 500 steps
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500

; Neighbor searching
nstlist         = 1
cutoff-scheme   = Verlet
ns_type         = grid

; Electrostatics / VdW
coulombtype     = cutoff
rcoulomb        = 1.0
rvdw            = 1.0

; No periodic boundary conditions (vacuum)
pbc             = no
"""


def _slurm_script(pdb_id: str, remote_dir: str) -> str:
    return f"""\
#!/bin/bash
#SBATCH --job-name=peat_{pdb_id}_em
#SBATCH --output={remote_dir}/peat_{pdb_id}_em_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition={_HPC_PARTITION}

set -e
cd {remote_dir}

module purge
module load {_GROMACS_MODULE}

echo "[PEAT] pdb2gmx"
gmx pdb2gmx \\
    -f {pdb_id}.pdb \\
    -o processed.gro \\
    -p topol.top \\
    -ff oplsaa \\
    -water none \\
    -ignh \\
    -nobackup 2>&1

echo "[PEAT] editconf"
gmx editconf \\
    -f processed.gro \\
    -o box.gro \\
    -c -d 1.0 -bt cubic \\
    -nobackup 2>&1

echo "[PEAT] grompp"
gmx grompp \\
    -f em.mdp \\
    -c box.gro \\
    -p topol.top \\
    -o em.tpr \\
    -maxwarn 5 \\
    -nobackup 2>&1

echo "[PEAT] mdrun"
gmx mdrun \\
    -v -deffnm em \\
    -ntmpi 1 -ntomp 4 \\
    -nobackup 2>&1

echo "[PEAT] done"
"""


def submit_minimization(pdb_id: str) -> dict:
    """
    Full energy minimization workflow for a PDB ID on Anvil HPC.

    Steps:
      1. Download PDB from RCSB to /tmp/peat_structures/ (cached).
      2. Write em.mdp and SLURM script to a local staging directory.
      3. SSH mkdir the remote working directory.
      4. SCP PDB, mdp, and SLURM script to HPC.
      5. SSH sbatch, parse and return the job ID.

    Returns:
      {"job_id": str, "remote_dir": str, "pdb_id": str}
    """
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured — cannot submit to HPC.")

    pdb_id = pdb_id.upper()
    import requests as _req  # local import to avoid circular issues

    # ── 1. Download PDB ───────────────────────────────────────────────────────
    os.makedirs(_CACHE_DIR, exist_ok=True)
    pdb_path = os.path.join(_CACHE_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        r = _req.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        r.raise_for_status()
        with open(pdb_path, "wb") as fh:
            fh.write(r.content)

    # ── 2. Write local staging files ──────────────────────────────────────────
    stage_dir = os.path.join(_CACHE_DIR, f"{pdb_id}_em")
    os.makedirs(stage_dir, exist_ok=True)

    mdp_path   = os.path.join(stage_dir, "em.mdp")
    remote_dir = f"{_HPC_WORKDIR}/{pdb_id}_em"

    with open(mdp_path, "w") as fh:
        fh.write(_EM_MDP)

    slurm_path = os.path.join(stage_dir, "run_em.sh")
    with open(slurm_path, "w") as fh:
        fh.write(_slurm_script(pdb_id, remote_dir))

    # ── 3. Create remote directory ────────────────────────────────────────────
    _ssh(f"mkdir -p {remote_dir}")

    # ── 4. SCP files to HPC ───────────────────────────────────────────────────
    _scp_to(pdb_path,    f"{remote_dir}/{pdb_id}.pdb")
    _scp_to(mdp_path,    f"{remote_dir}/em.mdp")
    _scp_to(slurm_path,  f"{remote_dir}/run_em.sh")

    # ── 5. Submit job ─────────────────────────────────────────────────────────
    out = _ssh(f"sbatch {remote_dir}/run_em.sh", workdir=remote_dir)
    # sbatch prints: "Submitted batch job 1234567"
    match = re.search(r"(\d+)", out)
    if not match:
        raise RuntimeError(f"sbatch did not return a job ID. Output:\n{out}")
    job_id = match.group(1)

    return {"job_id": job_id, "remote_dir": remote_dir, "pdb_id": pdb_id}


def check_job_status(job_id: str) -> str:
    """Run squeue -j {job_id} on HPC and return formatted status."""
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured.")
    out = _ssh(
        f"squeue -j {job_id} "
        f"--format='%.10i %.20j %.8T %.10M %.10l %R' --noheader 2>&1 "
        f"|| sacct -j {job_id} --format=JobID,JobName,State,Elapsed --noheader 2>&1"
    )
    return out.strip() or f"Job {job_id} not found in queue (may have completed)."


def download_job_results(job_id: str, pdb_id: str) -> dict:
    """
    SCP minimization output files from HPC back to
    /tmp/peat_structures/{pdb_id}_em_results/.

    Returns:
      {"local_dir": str, "files": list[str]}
    """
    if not _HPC_HOST:
        raise RuntimeError("HPC_HOST is not configured.")

    pdb_id    = pdb_id.upper()
    local_dir = os.path.join(_CACHE_DIR, f"{pdb_id}_em_results")
    remote_dir = f"{_HPC_WORKDIR}/{pdb_id}_em"

    os.makedirs(local_dir, exist_ok=True)

    # Download em.* (em.gro, em.log, em.edr, em.trr) and the SLURM log
    _scp_from(f"{remote_dir}/em.*",                    local_dir)
    _scp_from(f"{remote_dir}/peat_{pdb_id}_em_{job_id}.log", local_dir)

    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    return {"local_dir": local_dir, "files": sorted(files)}
