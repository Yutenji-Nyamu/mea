You are TaskGen for a minimal LIBERO experiment.
Write a complete BDDL problem derived from the official source below.

The planner proposal was produced only after an official control rollout:
{
  "schema_version": 1,
  "action": "continue",
  "sub_aspect": "object_identity.goal_language_dependency",
  "hypothesis": "SmolVLA's failure in the official control task is due to a dependency on the specific language phrasing of the goal.",
  "requested_perturbation": {
    "description": "Modify the goal language while preserving its semantic meaning.",
    "controlled_changes": [
      "language"
    ],
    "preserve": [
      "task identity",
      "policy checkpoint"
    ]
  },
  "task_need": {
    "required": true,
    "description": "TaskGen must edit the goal language in the BDDL while ensuring the task remains semantically identical."
  },
  "tool_need": {
    "required": true,
    "description": "ToolGen must retrieve the success metric for the modified task execution.",
    "reuse_first": true
  },
  "rationale": "The failure in the official control task suggests a potential issue with SmolVLA's ability to generalize to variations in task representation. Testing a language modification isolates whether the policy is overly reliant on specific goal phrasing, which is a critical sub-aspect of robustness to object identity changes."
}

Phase-1 compatibility contract:
- Keep the problem name/domain, fixtures, regions, objects, and every :init
  predicate exactly unchanged.
- Change only :language, :obj_of_interest, and :goal.
- Select exactly one existing non-basket object other than alphabet_soup_1.
- The sole goal must be `(In <selected_object> basket_1_contain_region)`.
- The language and obj_of_interest must name the same selected object.
- Do not add objects, regions, comments, code, or Markdown.

Original Query:
How robust is SmolVLA to task-relevant object identity changes in LIBERO, and where does it first fail?

OFFICIAL BDDL:
(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick the alphabet soup and place it in the basket)
    (:regions
      (bin_region
          (:target floor)
          (:ranges (
              (-0.01 0.25 0.01 0.27)
            )
          )
      )
      (target_object_region
          (:target floor)
          (:ranges (
              (-0.145 -0.265 -0.095 -0.215)
            )
          )
      )
      (other_object_region_0
          (:target floor)
          (:ranges (
              (0.025 -0.125 0.07500000000000001 -0.07500000000000001)
            )
          )
      )
      (other_object_region_1
          (:target floor)
          (:ranges (
              (-0.175 0.034999999999999996 -0.125 0.08499999999999999)
            )
          )
      )
      (other_object_region_2
          (:target floor)
          (:ranges (
              (0.07500000000000001 -0.225 0.125 -0.17500000000000002)
            )
          )
      )
      (other_object_region_3
          (:target floor)
          (:ranges (
              (0.125 0.0049999999999999975 0.175 0.055)
            )
          )
      )
      (other_object_region_4
          (:target floor)
          (:ranges (
              (-0.225 -0.10500000000000001 -0.17500000000000002 -0.055)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    floor - floor
  )

  (:objects
    alphabet_soup_1 - alphabet_soup
    basket_1 - basket
    salad_dressing_1 - salad_dressing
    cream_cheese_1 - cream_cheese
    milk_1 - milk
    tomato_sauce_1 - tomato_sauce
    butter_1 - butter
  )

  (:obj_of_interest
    alphabet_soup_1
    basket_1
  )

  (:init
    (On alphabet_soup_1 floor_target_object_region)
    (On salad_dressing_1 floor_other_object_region_0)
    (On cream_cheese_1 floor_other_object_region_1)
    (On milk_1 floor_other_object_region_2)
    (On tomato_sauce_1 floor_other_object_region_3)
    (On butter_1 floor_other_object_region_4)
    (On basket_1 floor_bin_region)
  )

  (:goal
    (And (In alphabet_soup_1 basket_1_contain_region))
  )

)


Return strict JSON with exactly:
{"bddl_text":"the complete BDDL text","selected_object":"object_id","rationale":"short reason"}
