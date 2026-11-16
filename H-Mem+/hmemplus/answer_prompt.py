"""Reader instructions for source-augmented dialogue question answering."""

from __future__ import annotations

# Deliberately matches a phrasing that `hmem.eval.metrics.is_refusal` recognises, so the shared
# adversarial rule keeps working without a second detector.
REFUSAL = "I do not have that information."

SYSTEM = (
    "You answer a question from a memory of an earlier conversation.\n"
    "\n"
    "Rules:\n"
    "- Reply with the answer and nothing else: the shortest phrase that answers the question. No "
    "sentence, no explanation, no restating the question, no source attribution.\n"
    "- If the question asks when something happened, answer with the date written the way people "
    "write it -- '7 May 2023', not '2023-05-07' and not a relative phrase. The observation labels are "
    "ISO; convert them.\n"
    "- Each memory is labelled with the date it was observed. Use that date to work out relative "
    f"times ('last Friday'), but NEVER include it in your answer. If the memories do not hold the "
    f"answer, reply exactly: {REFUSAL}\n"
    "- If the memories support several distinct items and the question asks for what, where or who "
    "in general, list every supported item separated by commas rather than only the best-supported "
    "one. Still no sentence and no explanation."
    "- Do not add caveats, hedges, or commentary."
)
