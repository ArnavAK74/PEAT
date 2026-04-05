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
import stat
import subprocess
import shlex
import tempfile

# ── Config ────────────────────────────────────────────────────────────────────
_ALLOWED_PREFIXES = {
    "gmx", "python", "python3", "bash", "sh",
    "squeue", "sacct", "sbatch", "scancel",
    "echo", "ls", "cat", "head", "tail",
}
_TIMEOUT = int(os.getenv("HPC_TIMEOUT_SECONDS", "300"))
_MAX_OUTPUT_CHARS = 4000

# SSH settings (prod)
_HPC_HOST    = os.getenv("HPC_HOST", "")
_HPC_USER    = os.getenv("HPC_USER", "")
_HPC_WORKDIR = os.getenv("HPC_WORKDIR", "~/peat_runs")


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


def _run_local(cmd: str) -> str:
    """Run the command in a local subprocess."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = f"ERROR: command timed out after {_TIMEOUT} s."
    except Exception as e:
        output = f"ERROR: {e}"
    return output[:_MAX_OUTPUT_CHARS]


def _run_remote(cmd: str) -> str:
    """
    SSH into HPC_HOST and run the command in HPC_WORKDIR.

    Reads the private key from the HPC_PRIVATE_KEY environment variable,
    writes it to a temporary file with mode 600, uses it for authentication,
    then deletes it immediately after the command finishes.
    """
    private_key = os.getenv("HPC_PRIVATE_KEY", "")
    if not private_key:
        return "ERROR: HPC_PRIVATE_KEY env var is not set."

    # Replace literal \n with real newlines (common when set via env/secrets)
    private_key = private_key.replace("\\n", "\n")
    # Ensure the key ends with a newline (required by OpenSSH)
    if not private_key.endswith("\n"):
        private_key += "\n"

    print(f"[HPC] key first 20: {repr(private_key[:20])}")
    print(f"[HPC] key last  20: {repr(private_key[-20:])}")

    key_file = None
    try:
        # Write key to a secure temp file
        fd, key_path = tempfile.mkstemp(prefix="peat_hpc_", suffix=".pem")
        key_file = key_path
        os.write(fd, private_key.encode())
        os.close(fd)
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)  # 600

        ssh_cmd = (
            f"ssh "
            f"-i {key_path} "
            f"-o StrictHostKeyChecking=no "
            f"-o BatchMode=yes "
            f"{_HPC_USER}@{_HPC_HOST} "
            f"'cd {_HPC_WORKDIR} && {cmd}'"
        )
        return _run_local(ssh_cmd)
    finally:
        if key_file and os.path.exists(key_file):
            os.remove(key_file)
