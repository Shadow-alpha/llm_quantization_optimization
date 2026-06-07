def greedy_bit_assignment(costs, sizes, candidate_bits, budget_mb):
    assignment = {name: max(candidate_bits) for name in costs}

    def memory_mb(assign):
        return sum(sizes[name] * bits for name, bits in assign.items()) / 8 / 1024 / 1024

    while memory_mb(assignment) > budget_mb:
        best = None
        for name, bits in assignment.items():
            lower = [b for b in candidate_bits if b < bits]
            if not lower:
                continue
            next_bits = max(lower)
            saved = sizes[name] * (bits - next_bits)
            penalty = costs[name][next_bits] - costs[name][bits]
            ratio = penalty / max(saved, 1)
            if best is None or ratio < best[0]:
                best = (ratio, name, next_bits)
        if best is None:
            break
        _, name, next_bits = best
        assignment[name] = next_bits
    return assignment

