def generated_tool(trajectory):
    left = np.asarray(trajectory.trace['roller_position'], dtype=float)
    right = np.asarray(trajectory.trace['non_target_roller_position'], dtype=float)
    terminal_index = len(left) - 1
    left_value = float(left[terminal_index, 2])
    right_value = float(right[terminal_index, 2])
    finite = bool(np.isfinite(left_value) and np.isfinite(right_value))
    signed_difference = left_value - right_value
    value = signed_difference if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {
        "value": value,
        "unit": 'm',
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {
            "operation": 'terminal_signal_difference',
            "left_signal": 'roller_position',
            "right_signal": 'non_target_roller_position',
            "component": 'z',
            "absolute": False,
            "left_terminal_value": left_value if finite else None,
            "right_terminal_value": right_value if finite else None,
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        },
    }
