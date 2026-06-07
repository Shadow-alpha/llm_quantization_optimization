from models import estimate_parameter_memory_mb


def model_memory_row(model, method: str, bit_overrides=None):
    return {"method": method, "memory_mb": estimate_parameter_memory_mb(model, bit_overrides)}

