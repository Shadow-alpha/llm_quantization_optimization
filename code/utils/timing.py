from evaluate import measure_latency_ms


def latency_row(model, batches, method: str):
    return {"method": method, "latency_ms": measure_latency_ms(model, batches)}

