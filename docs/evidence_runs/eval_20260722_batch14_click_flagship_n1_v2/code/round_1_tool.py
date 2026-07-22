def generated_tool(trajectory):
    bell_position = trajectory.trace["bell_position"]
    bell_contact_position = trajectory.trace["bell_contact_position"]
    physics_step = trajectory.trace["physics_step"]
    simulation_time_seconds = trajectory.trace["simulation_time_seconds"]

    if float(bell_position[0, 0]) < 0:
        active_arm = "left"
        tcp_position = trajectory.trace["left_tcp_position"]
    else:
        active_arm = "right"
        tcp_position = trajectory.trace["right_tcp_position"]

    d = np.sqrt(
        (tcp_position[:, 0] - bell_contact_position[:, 0]) ** 2
        + (tcp_position[:, 1] - bell_contact_position[:, 1]) ** 2
    )
    minimum_index = int(
        np.argmin(np.where(np.isfinite(d), d, np.inf))
    )
    minimum_error = d[minimum_index]

    if not np.isfinite(minimum_error):
        return {
            "value": None,
            "unit": "m",
            "passed": None,
            "evidence_steps": [],
            "details": {
                "active_arm": active_arm,
                "min_error_physics_step": None,
                "simulation_time_seconds": None,
            },
        }

    return {
        "value": float(minimum_error),
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_step[minimum_index])],
        "details": {
            "active_arm": active_arm,
            "min_error_physics_step": int(physics_step[minimum_index]),
            "simulation_time_seconds": float(
                simulation_time_seconds[minimum_index]
            ),
        },
    }
