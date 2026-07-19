def generated_tool(trajectory):
    hammer_z = trajectory.trace["hammer_position"][:, 2]
    initial_z = float(hammer_z[0])
    threshold = float(
        trajectory.schema.get("pickup_height_threshold_m", 0.03)
    )
    rise = hammer_z - initial_z
    pickup_indices = np.where(rise >= threshold)[0]
    pickup_index = int(pickup_indices[0]) if len(pickup_indices) else None
    pickup_step = (
        int(trajectory.trace["physics_step"][pickup_index])
        if pickup_index is not None
        else None
    )
    pickup_time = (
        float(trajectory.trace["simulation_time_seconds"][pickup_index])
        if pickup_index is not None
        else None
    )

    contacts = [
        item
        for item in trajectory.hammer_block_contacts()
        if item.get("physical_contact", False)
    ]
    first_contact = (
        min(
            contacts,
            key=lambda item: item["first_physical_physics_step"],
        )
        if contacts
        else None
    )
    contact_step = (
        int(first_contact["first_physical_physics_step"])
        if first_contact is not None
        else None
    )
    contact_time = (
        float(first_contact["first_physical_simulation_time_seconds"])
        if first_contact is not None
        else None
    )

    pickup_detected = pickup_index is not None
    contact_detected = first_contact is not None
    ordering_valid = (
        pickup_detected
        and contact_detected
        and contact_time >= pickup_time
    )
    contact_precedes_pickup = (
        pickup_detected
        and contact_detected
        and contact_time < pickup_time
    )

    reason = (
        "pickup_not_observed"
        if not pickup_detected
        else (
            "contact_not_observed_after_pickup"
            if not contact_detected
            else (
                "contact_precedes_pickup"
                if contact_precedes_pickup
                else "measured"
            )
        )
    )
    duration_seconds = (
        contact_time - pickup_time if ordering_valid else None
    )
    duration_physics_steps = (
        contact_step - pickup_step if ordering_valid else None
    )

    return {
        "value": duration_seconds,
        "unit": "s",
        "passed": None,
        "evidence_steps": sorted(
            [
                step
                for step in [pickup_step, contact_step]
                if step is not None
            ]
        ),
        "details": {
            "pickup_detected": pickup_detected,
            "contact_detected": contact_detected,
            "ordering_valid": ordering_valid,
            "pickup_physics_step": pickup_step,
            "contact_physics_step": contact_step,
            "pickup_time_seconds": pickup_time,
            "contact_time_seconds": contact_time,
            "duration_physics_steps": duration_physics_steps,
            "pickup_height_threshold_m": threshold,
            "reason": reason,
        },
    }
