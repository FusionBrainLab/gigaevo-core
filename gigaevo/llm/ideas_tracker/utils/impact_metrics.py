def avg_score(data: list[float]) -> float:
    return sum(data) / len(data)


def delta_impact(data1: list[float], data2: list[float]) -> float:
    return avg_score(data1) - avg_score(data2)
