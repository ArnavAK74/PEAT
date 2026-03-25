from Bio.PDB import PDBParser
import numpy as np


def build_3dmol_html(pdb_id: str) -> str:
    """
    Returns HTML string for embedding a 3Dmol.js viewer of the PDB entry.
    """
    return f"""
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<div id="viewer3d" style="width:100%; height:500px;"></div>
<script>
  const elt = document.getElementById('viewer3d');
  const viewer = $3Dmol.createViewer(elt, {{ backgroundColor: 'white' }});
  $3Dmol.download('pdb:{pdb_id.upper()}', viewer, {{}}, function() {{
    viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum' }} }});
    viewer.zoomTo();
    viewer.render();
  }});
</script>
"""  # noqa: E501


def find_hotspots(pdb_path: str, contact_threshold: int = 30, distance_cutoff: float = 6.0) -> list[str]:
    """
    Identifies residues with more than contact_threshold atoms within distance_cutoff Å.
    Returns list of residue identifiers (e.g. 'A123').
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_path)
    atoms = list(structure.get_atoms())
    hotspots = []
    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' not in res:
                    continue
                ca = res['CA'].get_vector()
                count = sum(
                    1 for atom in atoms
                    if (atom.get_vector() - ca).norm() < distance_cutoff
                )
                if count > contact_threshold:
                    hotspots.append(f"{chain.id}{res.id[1]}")
    return sorted(hotspots)