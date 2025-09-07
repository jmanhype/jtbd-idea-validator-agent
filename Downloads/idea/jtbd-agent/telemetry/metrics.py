def summarize(metrics: dict) -> str:
    return f"ΔScore={metrics['delta']:+.2f}  AKR={metrics['akr']:.2f}"
