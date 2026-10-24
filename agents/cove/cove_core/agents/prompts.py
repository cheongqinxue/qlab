"""CoVe prompt templates (paper-style four-step loop)."""

BASELINE_SYSTEM = (
    "You are a helpful assistant. Answer the user's question clearly and concisely."
)

BASELINE_HUMAN = "User question:\n{query}"

PLAN_SYSTEM = (
    "You identify factual claims that should be verified. "
    "Propose short, self-contained verification questions that check those facts."
)

PLAN_HUMAN = (
    "User question:\n{query}\n\n"
    "Draft response:\n{baseline}\n\n"
    "List verification questions to fact-check the draft."
)

EXECUTE_SYSTEM = (
    "Answer the factual question accurately and concisely. "
    "Do not mention or rely on any draft or unverified response."
)

EXECUTE_HUMAN = "{question}"

REVISE_SYSTEM = (
    "You revise a draft response using verified question–answer pairs. "
    "Fix factual errors; keep the answer helpful and concise."
)

REVISE_HUMAN = (
    "User question:\n{query}\n\n"
    "Draft response:\n{baseline}\n\n"
    "Verification results:\n{pairs_text}\n\n"
    "Write the final verified response."
)
