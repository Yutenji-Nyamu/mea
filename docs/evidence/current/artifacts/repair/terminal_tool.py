def generated_tool(trajectory):
    signal = np.asarray(trajectory.trace['bottle_functional_position'], dtype=float)
    terminal_index = len(signal) - 1
    raw_value = float(signal[terminal_index, 2])
    finite = bool(np.isfinite(raw_value))
    value = raw_value if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {
        "value": value,
        "unit": 'm',
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {
            "operation": 'terminal_signal_component',
            "signal": 'bottle_functional_position',
            "component": 'z',
            "absolute": False,
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        },
    }
