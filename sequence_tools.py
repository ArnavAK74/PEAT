from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align import MultipleSeqAlignment
from collections import Counter


def run_blast(sequence: str, program: str = "blastp", database: str = "nr"):
    """
    Runs NCBI BLAST for the input sequence and returns the parsed record.
    """
    result_handle = NCBIWWW.qblast(program, database, sequence)
    return NCBIXML.read(result_handle)


def conservation_scores(msa: MultipleSeqAlignment) -> list[float]:
    """
    Computes per-residue conservation scores from an MSA.
    """
    length = msa.get_alignment_length()
    scores = []
    for i in range(length):
        col = [rec.seq[i] for rec in msa]
        freq = Counter(col)
        top_count = freq.most_common(1)[0][1]
        scores.append(top_count / len(col))
    return scores