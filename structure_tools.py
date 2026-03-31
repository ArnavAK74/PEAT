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
