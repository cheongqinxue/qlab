import pytest
import re

@pytest.fixture
def sentences():
    return [
        "i am a happy man.", "i love my happy wife.", "i love to cook!", 
        "i love cooking for my wife.", "my wife loves me."
    ]

@pytest.fixture
def config():
    return {
        "ngram": 5,
        "target": 50,
        "eps": 0.001,
        "drop_rate": 0.05
    }

def test_preprocess(sentences, config):
    from unigram import preprocess
    cost, new_sentences = preprocess(sentences=sentences, n=config["ngram"])
    space_pattern = re.compile(" ")
    assert all([len(space_pattern.findall(sentence))==0 for sentence in new_sentences]), \
        f"sentence: {sentences[0]}"

    tokens = set(cost.keys())
    lengths = [len(t) for t in cost]
    assert min(lengths) == 1
    assert max(lengths) == config["ngram"]
    assert "!" in tokens
    assert new_sentences[0][-5:] in tokens

def test_pruning(sentences, config):
    from unigram import preprocess, pruning
    cost, new_sents = preprocess(sentences, config["ngram"])
    start_token_count = len(cost)
    new_cost = pruning(sentences, cost, 5)
    end_token_count = len(new_cost)
    assert start_token_count - end_token_count == 5, f"Delta: {start_token_count - end_token_count}"

def test_build(sentences, config):
    from unigram import build
    vocab_cost = build(sentences=sentences, **config)
    assert len(vocab_cost)==config["target"]

    print(vocab_cost.keys())
