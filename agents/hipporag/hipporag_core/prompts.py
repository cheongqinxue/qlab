"""OpenIE prompt shapes (aligned with upstream HippoRAG templates)."""

NER_PASSAGE_PROMPT = """Extract named entities and noun phrases from the passage.
Return JSON with key "entities" as a list of strings."""

NER_QUERY_PROMPT = """Extract named entities and noun phrases from the question.
Return JSON with key "entities" as a list of strings."""

TRIPLE_PROMPT = """Extract OpenIE triples (subject, predicate, object) from the passage.
Use entities as hints: {entities}
Return JSON with key "triples" as a list of objects with subject, predicate, object."""
