def generated_tool(trajectory):
    left_positions = trajectory.trace["left_tcp_position"]
    stapler_positions = trajectory.trace["stapler_position"]
    physics_steps = trajectory.trace["physics_step"]

    sample_count = min(len(left_positions), len(stapler_positions), len(physics_steps))

    for index in range(sample_count - 1, -1, -1):
        left_x = float(left_positions[index][0])
        left_y = float(left_positions[index][1])
        left_z = float(left_positions[index][2])
        stapler_x = float(stapler_positions[index][0])
        stapler_y = float(stapler_positions[index][1])
        stapler_z = float(stapler_positions[index][2])

        if (
            np.isfinite(left_x)
            and np.isfinite(left_y)
            and np.isfinite(left_z)
            and np.isfinite(stapler_x)
            and np.isfinite(stapler_y)
            and np.isfinite(stapler_z)
        ):
            dx = left_x - stapler_x
            dy = left_y - stapler_y
            dz = left_z - stapler_z
            distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            return {
                "value": distance,
                "unit": "m",
                "passed": None,
                "evidence_steps": [int(physics_steps[index])],
                "details": {
                    "operation": "derived_observable",
                    "reason": "measured",
                },
            }

    return {
        "value": None,
        "unit": "m",
        "passed": None,
        "evidence_steps": [],
        "details": {
            "operation": "derived_observable",
            "reason": "no_finite_sample",
        },
    }
