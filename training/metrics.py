from __future__ import annotations

from typing import Sequence, Tuple


def edit_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> Tuple[int, int, int]:
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    table = [[(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitute = table[row - 1][column - 1]
            delete = table[row - 1][column]
            insert = table[row][column - 1]
            table[row][column] = min(
                [
                    (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
                ],
                key=lambda item: item[0],
            )
    _, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions


def corpus_error_rate(
    references: Sequence[str], hypotheses: Sequence[str], unit: str
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis counts do not match.")
    errors = total = 0
    for reference, hypothesis in zip(references, hypotheses):
        if unit == "word":
            ref_tokens, hyp_tokens = reference.split(), hypothesis.split()
        elif unit == "char":
            ref_tokens = list(reference.replace(" ", ""))
            hyp_tokens = list(hypothesis.replace(" ", ""))
        else:
            raise ValueError("unit must be 'word' or 'char'.")
        substitutions, deletions, insertions = edit_counts(ref_tokens, hyp_tokens)
        errors += substitutions + deletions + insertions
        total += len(ref_tokens)
    return errors / max(total, 1)
