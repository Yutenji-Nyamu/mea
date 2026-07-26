"""Generated predicate Tool for one LIBERO evaluation."""

def evaluate_episode(record):
    if "goal_predicate_satisfied" not in record:
        raise ValueError("missing goal_predicate_satisfied")
    value = record["goal_predicate_satisfied"]
    if not isinstance(value, bool):
        raise TypeError("goal_predicate_satisfied must be boolean")
    return {
        "tool": "libero_goal_predicate_tool",
        "value": value,
        "unit": None,
        "passed": value,
        "evidence_steps": [max(0, int(record.get("executed_steps", 1)) - 1)],
        "details": {"predicate": record.get("goal_predicates", [])},
    }
