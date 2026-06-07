def pareto_frontier(points, x_key="memory_mb", y_key="perplexity"):
    sorted_points = sorted(points, key=lambda p: (p[x_key], p[y_key]))
    frontier = []
    best_y = float("inf")
    for point in sorted_points:
        if point[y_key] < best_y:
            frontier.append(point)
            best_y = point[y_key]
    return frontier

