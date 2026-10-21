from math import log

from vertibi.vertibi import vertibi


def preprocess(sentences: list[str], n: int) -> tuple[dict[str, float], list[str]]:
    vocabulary: dict[str, int] = {}
    sentences = [sentence.replace(" ", "_") for sentence in sentences]
    for sentence in sentences:
        total_count = 0
        for ngram in range(1, n + 1):
            for start in range(0, len(sentence) - ngram + 1):
                end = start + ngram
                sub = sentence[start:end]
                if not sub in vocabulary:
                    vocabulary[sub] = 1
                else:
                    vocabulary[sub] += 1
                total_count += 1
    cost = {k: -log(v / total_count) for k, v in vocabulary.items()}
    return cost, sentences


def decode_and_score(
    sentences: list[str], vocab_cost: dict[str, float]
) -> tuple[dict[str, int], int, float]:
    counts: dict[str, int] = {k: 0 for k in vocab_cost}
    total_count = 0
    total_score = 0
    for sentence in sentences:
        tokens, score = vertibi(raw_string=sentence, vocab_cost=vocab_cost)
        for token in tokens:
            counts[token] += 1
            total_count += 1
        total_score += score

    return counts, total_count, total_score


def em(
    sentences: list[str], vocab_cost: dict[str, float], eps: float = 1e-5
) -> dict[str, float]:
    counts, total_count, _ = decode_and_score(
        sentences=sentences, vocab_cost=vocab_cost
    )
    for k, v in counts.items():
        if v == 0:
            v = eps
        vocab_cost[k] = -log(v / total_count)
    return vocab_cost


def pruning(
    sentences: list[str], vocab_cost: dict[str, float], drop_count: int
) -> dict[str, float]:
    counts, total_count, base_score = decode_and_score(
        sentences=sentences, vocab_cost=vocab_cost
    )
    token_loss = {k: float("inf") for k in vocab_cost}
    for token in counts:
        trunc_vocab_score = {k: v for k, v in vocab_cost.items() if k != token}
        _, _, new_score = decode_and_score(
            sentences=sentences, vocab_cost=trunc_vocab_score
        )
        token_loss[token] = base_score - new_score

    token_loss = dict(sorted(token_loss.items(), key=lambda x: x[1], reverse=True))
    tokens = list(token_loss.keys())
    items_to_remove = set()
    i = 0
    while len(items_to_remove) < drop_count:
        if len(tokens[i]) > 1:
            items_to_remove.add(tokens[i])
        i += 1
    return {k: v for k, v in vocab_cost.items() if k not in items_to_remove}


def build(sentences: list[str], ngram: int, eps: float, target: int, drop_rate: float):
    vocab_cost, sentences = preprocess(sentences, n=ngram)
    while True:
        drop_to_target = max(int((1 - drop_rate) * len(vocab_cost)), target)
        drop_count = len(vocab_cost) - drop_to_target
        vocab_cost = em(sentences=sentences, vocab_cost=vocab_cost, eps=eps)
        vocab_cost = pruning(
            sentences=sentences, vocab_cost=vocab_cost, drop_count=drop_count
        )
        if len(vocab_cost) <= target:
            break
    return vocab_cost
