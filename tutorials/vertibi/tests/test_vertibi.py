import pytest


def test_vertibi():
    from vertibi import vertibi

    string:str="banana apple pineapple"
    vocab_cost:dict[str,float]={
        "na":9,
        "ba":9,
        "_ba":8,
        "banana":8,
        "_banana":7,
        "apple":7,
        "_apple":6,
        "pine":7,
        "_pine":6,
        "pineapple":11,
        "_pineapple":15
    }

    (tokens, score) = vertibi(raw_string=string, vocab_cost=vocab_cost)

    answer = ["banana", "_apple", "_pine", "apple"]

    assert tokens==answer, f"Got: {tokens}"
    assert score==sum([vocab_cost[a] for a in answer]), f"Scored: {score}"