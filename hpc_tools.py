# hpc_tools.py  (v3 — Globus Compute for job submission, SSH for status queries)
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

_HPC_HOST      = os.getenv("HPC_HOST", "")
_HPC_USER      = os.getenv("HPC_USER", "")
_HPC_WORKDIR   = os.getenv("HPC_WORKDIR", "~/peat_runs")
_HPC_PARTITION = os.getenv("HPC_PARTITION", "gpu")
_HPC_QOS       = os.getenv("HPC_QOS", "gpu")
_HPC_ACCOUNT   = os.getenv("HPC_ACCOUNT", "")
_GROMACS_MODULE = os.getenv("GROMACS_MODULE", "gromacs/2024.1")
_GC_ENDPOINT   = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID", "")


# ── SSH key context manager ───────────────────────────────────────────────────
@contextmanager
def _key_file():
    private_key = os.getenv("HPC_PRIVATE_KEY", "")
    if not private_key:
        raise RuntimeError("HPC_PRIVATE_KEY env var is not set.")
    private_key = private_key.replace("\\n", "\n")
    if not private_key.endswith("\n"):
        private_key += "\n"
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


# ── Globus Compute client (lazy singleton, confidential client auth) ──────────
_gc_client = None

def _get_gc_client():
    global _gc_client
    if _gc_client is None:
        from globus_compute_sdk import Client
        from globus_sdk import ClientApp
        client_id     = os.getenv("GLOBUS_CLIENT_ID", "")
        client_secret = os.getenv("GLOBUS_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise RuntimeError("GLOBUS_CLIENT_ID and GLOBUS_CLIENT_SECRET must be set.")
        app = ClientApp(app_name="PEAT", client_id=client_id, client_secret=client_secret)
        _gc_client = Client(app=app)
    return _gc_client


# ── General HPC command ───────────────────────────────────────────────────────
def run_hpc_command(cmd: str) -> str:
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
        try:
            return _ssh(cmd, workdir=_HPC_WORKDIR)
        except RuntimeError as e:
            return f"ERROR: {e}"
    else:
        return _run_local(cmd)


# ── GROMACS energy minimization MDP ──────────────────────────────────────────
_EM_MDP = """\
; Steepest-descent energy minimization — vacuum, 500 steps
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
nstlist         = 1
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = cutoff
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = no
"""


# ── Globus Compute task function (runs on Anvil) ──────────────────────────────
def _run_minimization_on_hpc(pdb_id: str, pdb_content: str, workdir: str,
                              account: str, partition: str, qos: str,
                              gromacs_module: str, em_mdp: str) -> dict:
    """Executed remotely on Anvil via Globus Compute."""
    import os, subprocess, tempfile

    run_dir = os.path.join(workdir, f"{pdb_id}_em")
    os.makedirs(run_dir, exist_ok=True)

    pdb_path = os.path.join(run_dir, f"{pdb_id}.pdb")
    mdp_path = os.path.join(run_dir, "em.mdp")

    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    with open(mdp_path, "w") as f:
        f.write(em_mdp)

    gpu_line = "#SBATCH --gpus-per-node=1\n" if "gpu" in partition.lower() else ""
    account_line = f"#SBATCH --account={account}\n" if account else ""
    qos_line = f"#SBATCH --qos={qos}\n" if qos else ""

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=peat_{pdb_id}_em
#SBATCH --output={run_dir}/peat_{pdb_id}_em_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition={partition}
{qos_line}{account_line}{gpu_line}
set -e
cd {run_dir}
module purge
module load {gromacs_module}
gmx pdb2gmx -f {pdb_id}.pdb -o processed.gro -p topol.top -ff oplsaa -water none -ignh -nobackup 2>&1
gmx editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic -nobackup 2>&1
gmx grompp -f em.mdp -c box.gro -p topol.top -o em.tpr -maxwarn 5 -nobackup 2>&1
gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 4 -gpu_id 0 -nb gpu -bonded cpu -nobackup 2>&1
echo "[PEAT] done"
"""

    slurm_path = os.path.join(run_dir, "run_em.sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    result = subprocess.run(
        ["sbatch", slurm_path], capture_output=True, text=True, cwd=run_dir
    )
    output = result.stdout + result.stderr

    import re
    match = re.search(r"(\d+)", output)
    if not match:
        raise RuntimeError(f"sbatch failed:\n{output}")

    return {"job_id": match.group(1), "remote_dir": run_dir, "output": output}


# ── Public API: submit minimization ──────────────────────────────────────────
def submit_minimization(pdb_id: str) -> dict:
    if not _GC_ENDPOINT:
        raise RuntimeError("GLOBUS_COMPUTE_ENDPOINT_ID is not configured.")

    pdb_id = pdb_id.upper()
    import requests as _req

    os.makedirs(_CACHE_DIR, exist_ok=True)
    pdb_path = os.path.join(_CACHE_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        r = _req.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        r.raise_for_status()
        with open(pdb_path, "wb") as fh:
            fh.write(r.content)

    with open(pdb_path, "r") as fh:
        pdb_content = fh.read()

    gc = _get_gc_client()
    future = gc.run(
        _run_minimization_on_hpc,
        pdb_id, pdb_content, _HPC_WORKDIR,
        _HPC_ACCOUNT, _HPC_PARTITION, _HPC_QOS,
        _GROMACS_MODULE, _EM_MDP,
        endpoint_id=_GC_ENDPOINT,
    )

    # Wait up to 60s for sbatch to return a job ID
    result = future.result(timeout=60)
    return {
        "job_id":     result["job_id"],
        "remote_dir": result["remote_dir"],
        "pdb_id":     pdb_id,
    }


# ── Public API: check job status ──────────────────────────────────────────────
def check_job_status(job_id: str) -> str:
    """Check SLURM job via Globus Compute (no SSH needed)."""
    if not _GC_ENDPOINT:
        raise RuntimeError("GLOBUS_COMPUTE_ENDPOINT_ID is not configured.")

    def _check(job_id):
        import subprocess
        r = subprocess.run(
            ["squeue", "-j", job_id,
             "--format=%.10i %.20j %.8T %.10M %.10l %R", "--noheader"],
            capture_output=True, text=True,
        )
        out = r.stdout.strip()
        if not out:
            r2 = subprocess.run(
                ["sacct", "-j", job_id,
                 "--format=JobID,JobName,State,Elapsed", "--noheader"],
                capture_output=True, text=True,
            )
            out = r2.stdout.strip()
        return out or f"Job {job_id} not found in queue (may have completed)."

    gc = _get_gc_client()
    future = gc.run(_check, job_id, endpoint_id=_GC_ENDPOINT)
    return future.result(timeout=30)


# ── Public API: download results ──────────────────────────────────────────────
def download_job_results(job_id: str, pdb_id: str) -> dict:
    """Copy output files from Anvil via Globus Compute."""
    if not _GC_ENDPOINT:
        raise RuntimeError("GLOBUS_COMPUTE_ENDPOINT_ID is not configured.")

    pdb_id = pdb_id.upper()
    local_dir = os.path.join(_CACHE_DIR, f"{pdb_id}_em_results")
    os.makedirs(local_dir, exist_ok=True)
    remote_dir = f"{_HPC_WORKDIR}/{pdb_id}_em"

    def _read_files(remote_dir, pdb_id, job_id):
        import os, glob
        patterns = [
            os.path.join(remote_dir, "em.*"),
            os.path.join(remote_dir, f"peat_{pdb_id}_em_{job_id}.log"),
        ]
        files = {}
        for pattern in patterns:
            for path in glob.glob(pattern):
                with open(path, "rb") as f:
                    files[os.path.basename(path)] = f.read()
        return files

    gc = _get_gc_client()
    future = gc.run(_read_files, remote_dir, pdb_id, job_id, endpoint_id=_GC_ENDPOINT)
    remote_files = future.result(timeout=120)

    saved = []
    for name, content in remote_files.items():
        out_path = os.path.join(local_dir, name)
        with open(out_path, "wb") as f:
            f.write(content)
        saved.append(name)

    return {"local_dir": local_dir, "files": sorted(saved)}
