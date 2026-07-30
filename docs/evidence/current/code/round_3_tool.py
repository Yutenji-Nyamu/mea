def generated_tool(trajectory):
    right_tcp = trajectory.trace["right_tcp_position"]
    roller_contact = trajectory.trace["roller_right_contact_position"]
    physics_steps = trajectory.trace["physics_step"]
    deltas = right_tcp - roller_contact
    distances = np.sqrt(np.sum(deltas * deltas, axis=1))
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
    masked_distances = np.where(finite, distances, np.inf)
    index = int(np.argmin(masked_distances))
    return {
        "value": float(distances[index]),
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_steps[index])],
        "details": {
            "operation": "minimum_distance",
            "reason": "measured",
        },
    }
