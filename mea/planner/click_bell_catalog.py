"""Data-only click_bell retrieval metadata.

The production task/checkpoint catalog needs these capability descriptions,
but it must not import the historical click_bell planner implementations.
"""

CLICK_BELL_TEMPLATE_IDS = (
    "object_position.left_fixed",
    "object_position.right_fixed",
)
CLICK_BELL_POSITIONS = {
    "object_position.left_fixed": [-0.20, -0.08],
    "object_position.right_fixed": [0.20, -0.08],
}

CLICK_BELL_ADAPTIVE_ASPECTS = {
    "object_position": {
        "description": (
            "Generalization across safe left/right workspace positions while "
            "holding the official randomly sampled bell instance constant by seed."
        ),
        "template_ids": [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ],
    },
    "object_instance": {
        "description": (
            "Generalization across official bell base0/base1 instances while "
            "holding the official randomly sampled pose constant by seed."
        ),
        "template_ids": [
            "object_instance.base0",
            "object_instance.base1",
        ],
    },
    "robustness.distractor_avoidance": {
        "description": (
            "Robustness to one nearby physical look-alike bell. The generated "
            "scene/checker preserves the official target-contact predicate and "
            "adds a latched no-distractor-contact constraint."
        ),
        "template_ids": ["robustness.distractor_avoidance.lookalike_bell"],
    },
    "robustness.scene_clutter": {
        "description": (
            "Robustness to RoboTwin's simulator-native tabletop clutter while "
            "preserving the official bell pose, instance sampling, task logic, "
            "and ACT checkpoint."
        ),
        "template_ids": ["robustness.scene_clutter.official_table"],
    },
    "scene_background_texture": {
        "description": (
            "Generalization to RoboTwin's simulator-native unseen wall and "
            "table textures in eval mode while preserving the official bell, "
            "task logic, and ACT checkpoint."
        ),
        "template_ids": ["scene_background_texture.unseen"],
    },
    "scene_lighting": {
        "description": (
            "Generalization to RoboTwin's simulator-native per-episode random "
            "directional and point-light colors without temporal light flicker."
        ),
        "template_ids": ["scene_lighting.static_random"],
    },
    "performance.completion_time_stability": {
        "description": (
            "Completion-time stability across official click_bell ACT seeds. "
            "The trusted time_to_success Tool and deterministic Aggregate "
            "report success-conditioned mean and dispersion; N=1 is wiring "
            "only, while budgets 3 or 5 support a small stability estimate."
        ),
        "template_ids": ["performance.completion_time_stability.official"],
    },
}

CLICK_BELL_ADAPTIVE_TEMPLATES = {
    "object_position.left_fixed": {
        "aspect_id": "object_position",
        "probe_role": "sentinel",
        "description": "Safe fixed left-workspace position.",
    },
    "object_position.right_fixed": {
        "aspect_id": "object_position",
        "probe_role": "counterfactual",
        "description": "Mirrored safe right-workspace position.",
    },
    "object_instance.base0": {
        "aspect_id": "object_instance",
        "probe_role": "sentinel",
        "description": "Official larger white/black base0 bell instance.",
    },
    "object_instance.base1": {
        "aspect_id": "object_instance",
        "probe_role": "counterfactual",
        "description": "Official smaller blue/brown base1 bell instance.",
    },
    "robustness.distractor_avoidance.lookalike_bell": {
        "aspect_id": "robustness.distractor_avoidance",
        "probe_role": "sentinel",
        "description": (
            "One alternate official bell instance is placed 0.12 m from the "
            "target; success requires the correct-arm target press and forbids "
            "every latched distractor contact."
        ),
    },
    "robustness.scene_clutter.official_table": {
        "aspect_id": "robustness.scene_clutter",
        "probe_role": "sentinel",
        "description": (
            "Official click_bell scene plus simulator-generated physical "
            "tabletop distractors."
        ),
    },
    "scene_background_texture.unseen": {
        "aspect_id": "scene_background_texture",
        "probe_role": "sentinel",
        "description": (
            "RoboTwin random_background with clean_background_rate=0 and the "
            "unseen texture split selected by eval mode."
        ),
    },
    "scene_lighting.static_random": {
        "aspect_id": "scene_lighting",
        "probe_role": "sentinel",
        "description": (
            "RoboTwin random_light enabled with crazy_random_light_rate=0, "
            "yielding one static randomized light setup per episode."
        ),
    },
    "performance.completion_time_stability.official": {
        "aspect_id": "performance.completion_time_stability",
        "probe_role": "sentinel",
        "description": (
            "Unchanged official click_bell scene measured with the trusted "
            "first-success timestamp over the requested ACT seed budget."
        ),
    },
}

__all__ = [
    "CLICK_BELL_ADAPTIVE_ASPECTS",
    "CLICK_BELL_ADAPTIVE_TEMPLATES",
    "CLICK_BELL_POSITIONS",
    "CLICK_BELL_TEMPLATE_IDS",
]
