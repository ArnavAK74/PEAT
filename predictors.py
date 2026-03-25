import requests


def predict_ddg_dynamut(pdb_file: str, chain: str, resnum: int, mutation: str) -> dict:
    """
    Calls a DynaMut-like service to predict ΔΔG for a mutation.
    """
    url = 'https://dynamut-api.example.org/predict'
    with open(pdb_file, 'rb') as f:
        files = {'structure': f}
        data = {'chain': chain, 'resnum': resnum, 'mutation': mutation}
        r = requests.post(url, files=files, data=data)
    r.raise_for_status()
    return r.json()


def predict_mcsmp_pi(pdb_file: str, chain: str, resnum: int, mutation: str) -> dict:
    """
    Calls an mCSM-PPI-like service to predict binding changes upon mutation.
    """
    url = 'https://mcsmp-api.example.org/predict'
    with open(pdb_file, 'rb') as f:
        files = {'structure': f}
        data = {'chain': chain, 'resnum': resnum, 'mutation': mutation}
        r = requests.post(url, files=files, data=data)
    r.raise_for_status()
    return r.json()