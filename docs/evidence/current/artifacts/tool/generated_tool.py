def generated_tool(trajectory):
    bell_position = trajectory.trace["bell_position"]
    bell_contact_position = trajectory.trace["bell_contact_position"]
    active_arm = (
        "left"
        if float(bell_position[0, 0]) < 0.0
        else "right"
    )
    tcp_position = trajectory.trace[active_arm + "_tcp_position"]
    delta_xy = tcp_position[:, :2] - bell_contact_position[:, :2]
    distances = np.sqrt(np.sum(delta_xy * delta_xy, axis=1))
    minimum_index = np.argmin(
        np.where(np.isfinite(distances), distances, np.inf)
    )
    minimum_distance = distances[minimum_index]
    physics_step = trajectory.trace["physics_step"][minimum_index]
    simulation_time_seconds = trajectory.trace[
        "simulation_time_seconds"
    ][minimum_index]
    return {
        "value": float(minimum_distance) if np.isfinite(minimum_distance) else None,
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_step)] if np.isfinite(minimum_distance) else [],
        "details": {
            "active_arm": active_arm,
            "min_error_physics_step": int(physics_step)
            if np.isfinite(minimum_distance)
            else None,
            "simulation_time_seconds": float(simulation_time_seconds)
            if np.isfinite(minimum_distance)
            else None,
        },
    }
