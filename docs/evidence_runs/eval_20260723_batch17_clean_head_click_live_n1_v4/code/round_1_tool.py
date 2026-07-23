def generated_tool(trajectory):
    bell_position = trajectory.trace["bell_position"]
    bell_contact_position = trajectory.trace["bell_contact_position"]
    active_arm = (
        "left"
        if bell_position[0, 0] < 0
        else "right"
    )
    tcp_position = trajectory.trace[
        "left_tcp_position" if active_arm == "left" else "right_tcp_position"
    ]
    delta_xy = tcp_position[:, :2] - bell_contact_position[:, :2]
    d = np.sqrt(np.sum(delta_xy * delta_xy, axis=1))
    min_index = np.argmin(np.where(np.isfinite(d), d, np.inf))
    min_error = d[min_index]
    physics_step = trajectory.trace["physics_step"][min_index]
    simulation_time_seconds = trajectory.trace["simulation_time_seconds"][min_index]
    return {
        "value": float(min_error) if np.isfinite(min_error) else None,
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_step)] if np.isfinite(min_error) else [],
        "details": {
            "active_arm": active_arm,
            "min_error_physics_step": int(physics_step) if np.isfinite(min_error) else None,
            "simulation_time_seconds": (
                float(simulation_time_seconds) if np.isfinite(min_error) else None
            ),
        },
    }
