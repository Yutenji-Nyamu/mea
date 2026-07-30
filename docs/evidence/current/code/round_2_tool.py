def generated_tool(trajectory):
    left = trajectory.trace["left_tcp_position"]
    contact = trajectory.trace["roller_left_contact_position"]
    steps = trajectory.trace["physics_step"]
    delta = left - contact
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    finite = np.isfinite(distances)
    if not np.any(finite):
        return {
            "value": None,
            "unit": "m",
            "passed": None,
            "evidence_steps": [],
            "details": {
                "operation": "minimum_distance",
                "reason": "no_finite_sample",
            },
        }
    masked = np.where(finite, distances, np.inf)
    index = int(np.argmin(masked))
    return {
        "value": float(distances[index]),
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(steps[index])],
        "details": {
            "operation": "minimum_distance",
            "reason": "measured",
        },
    }
