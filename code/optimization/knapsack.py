def solve_layer_bit_knapsack(costs, sizes, candidate_bits, budget_mb):
    """Multiple-choice knapsack over layer bit-widths using MB scaled to KB states."""
    layers = list(costs)
    capacity_kb = int(budget_mb * 1024)
    dp = {0: (0.0, {})}

    for name in layers:
        next_dp = {}
        for used_kb, (loss, assign) in dp.items():
            for bits in candidate_bits:
                mem_kb = int(sizes[name] * bits / 8 / 1024)
                new_used = used_kb + mem_kb
                if new_used > capacity_kb:
                    continue
                new_loss = loss + costs[name][bits]
                if new_used not in next_dp or new_loss < next_dp[new_used][0]:
                    new_assign = dict(assign)
                    new_assign[name] = bits
                    next_dp[new_used] = (new_loss, new_assign)
        dp = next_dp or dp

    if not dp:
        return {name: min(candidate_bits) for name in layers}
    return min(dp.values(), key=lambda item: item[0])[1]

