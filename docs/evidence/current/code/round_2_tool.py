def generated_tool(trajectory):
    left = np.asarray(trajectory.trace['left_tcp_position'], dtype=float)
    right = np.asarray(trajectory.trace['bell_contact_position'], dtype=float)
    left_view = left[:, [0, 1]]
    right_view = right[:, [0, 1]]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not np.isfinite(value):
        return {
            "value": None,
            "unit": 'm',
            "passed": None,
            "evidence_steps": [],
            "details": {
                "operation": 'minimum_distance',
                "left_signal": 'left_tcp_position',
                "right_signal": 'bell_contact_position',
                "dimensions": ['x', 'y'],
                "min_index": None,
                "reason": "no_finite_sample",
            },
        }
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step = int(physics[index])
    return {
        "value": value,
        "unit": 'm',
        "passed": None,
        "evidence_steps": [step],
        "details": {
            "operation": 'minimum_distance',
            "left_signal": 'left_tcp_position',
            "right_signal": 'bell_contact_position',
            "dimensions": ['x', 'y'],
            "min_index": index,
            "reason": "measured",
        },
    }
