def build_3dmol_html(pdb_id: str, pdb_path: str | None = None) -> str:
    """
    Returns HTML string for embedding a 3Dmol.js viewer.

    If pdb_path is provided, the PDB file is read and embedded inline
    (used for AlphaFold structures not available on RCSB).
    Otherwise the structure is fetched from RCSB by pdb_id.
    """
    if pdb_path:
        with open(pdb_path) as fh:
            pdb_data = fh.read().replace("`", "\\`")  # escape backticks for JS template literal
        return f"""
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<div id="viewer3d" style="width:100%; height:500px;"></div>
<script>
  const elt = document.getElementById('viewer3d');
  const viewer = $3Dmol.createViewer(elt, {{ backgroundColor: 'white' }});
  const pdbData = `{pdb_data}`;
  viewer.addModel(pdbData, 'pdb');
  viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum' }} }});
  viewer.zoomTo();
  viewer.render();
</script>
"""  # noqa: E501

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
