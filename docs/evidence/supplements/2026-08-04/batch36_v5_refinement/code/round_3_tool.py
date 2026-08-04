def generated_tool(trajectory):
    left_tcp_positions = trajectory.trace["left_tcp_position"]
    stapler_positions = trajectory.trace["stapler_position"]
    physics_steps = trajectory.trace["physics_step"]

    terminal_index = len(left_tcp_positions) - 1
    left_position = left_tcp_positions[terminal_index]
    stapler_position = stapler_positions[terminal_index]

    if not bool(np.all(np.isfinite(left_position))) or not bool(np.all(np.isfinite(stapler_position))):
        return {
            "value": None,
            "unit": "m",
            "passed": None,
            "evidence_steps": [int(physics_steps[terminal_index])],
            "details": {
                "operation": "terminal_minimum_distance",
                "reason": "terminal_not_finite",
            },
        }

    distance = np.linalg.norm(left_position - stapler_position)
    return {
        "value": float(distance),
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_steps[terminal_index])],
        "details": {
            "operation": "terminal_minimum_distance",
            "reason": "measured",
        },
    }
