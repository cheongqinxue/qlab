def vertibi(raw_string:str, vocab_cost:dict[str,float])->tuple[list[str], float]:
    string:str = raw_string.replace(" ","_") 
    N = len(string)
    best_costs:list[float] = [float('inf')] * (N+1)
    best_costs[0] = 0
    backpointers:list[int] = [0] * (N+1)

    for end in range(1,N+1):
        for start in range(0,N):
            if best_costs[start] == float('inf'):
                continue
            substring:str = string[start:end] 
            token_cost:float|None = vocab_cost.get(substring)
            if token_cost is None:
                continue
            cost:float = best_costs[start] + token_cost
            if cost < best_costs[end]:
                best_costs[end] = cost
                backpointers[end] = start
    tokens = []
    curr = N
    while curr > 0:
        prev = backpointers[curr]
        tokens.append(string[prev:curr])
        curr = prev
    
    return tokens[::-1], best_costs[N]

