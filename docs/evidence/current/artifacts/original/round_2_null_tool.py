def generated_tool(trajectory):
    start_events = [
        item for item in trajectory.events
        if item.get('type') == 'contact_interval' and item.get('physical_contact') is True
    ]
    end_events = [
        item for item in trajectory.events
        if item.get('type') == 'success_transition'
    ]
    start_points = [
        (float(item.get('first_physical_simulation_time_seconds')), int(item.get('first_physical_physics_step')))
        for item in start_events
        if item.get('first_physical_simulation_time_seconds') is not None
        and item.get('first_physical_physics_step') is not None
    ]
    end_points = [
        (float(item.get('simulation_time_seconds')), int(item.get('physics_step')))
        for item in end_events
        if item.get('simulation_time_seconds') is not None
        and item.get('physics_step') is not None
    ]
    start = min(start_points) if start_points else None
    end = min(end_points) if end_points else None
    valid = bool(start is not None and end is not None and end[0] >= start[0])
    steps = sorted(list(set([
        item[1] for item in [start, end] if item is not None
    ])))
    if start is None:
        reason = "start_event_missing"
    elif end is None:
        reason = "end_event_missing"
    elif end[0] < start[0]:
        reason = "end_event_precedes_start_event"
    else:
        reason = "measured"
    return {
        "value": float(end[0] - start[0]) if valid else None,
        "unit": 's',
        "passed": None,
        "evidence_steps": steps,
        "details": {
            "operation": 'time_between_events',
            "start_event": {'actors': None, 'event_type': 'contact_interval', 'physical_only': True},
            "end_event": {'actors': None, 'event_type': 'success_transition', 'physical_only': False},
            "start_simulation_time_seconds": start[0] if start else None,
            "end_simulation_time_seconds": end[0] if end else None,
            "start_physics_step": start[1] if start else None,
            "end_physics_step": end[1] if end else None,
            "reason": reason,
        },
    }
