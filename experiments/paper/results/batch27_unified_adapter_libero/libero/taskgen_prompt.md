You are TaskGen for a minimal LIBERO experiment.
Write a complete BDDL problem derived from the official source below.

The planner proposal was produced only after an official control rollout:
{
  "schema_version": 1,
  "action": "continue",
  "sub_aspect": "goal_object_identity_change",
  "hypothesis": "The SmolVLA policy can successfully adapt to a change in the goal object identity while preserving the initial scene state.",
  "requested_perturbation": {
    "description": "Change the goal object identity to a different object while keeping the initial scene state unchanged.",
    "controlled_changes": [
      "goal_object_identity"
    ],
    "preserve": [
      "task identity",
      "policy checkpoint"
    ]
  },
  "task_need": {
    "required": true,
    "description": "TaskGen must modify the goal object identity in the task while ensuring compatibility with the initial scene state."
  },
  "tool_need": {
    "required": true,
    "description": "ToolGen must retrieve or generate a success metric to evaluate the policy's performance under the modified goal object identity.",
    "reuse_first": true
  },
  "rationale": "Testing the policy's ability to handle a change in goal object identity directly addresses the original Query and is the next logical step after confirming the policy's success in the unperturbed task."
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
Can this policy remain successful when the goal object identity changes while the initial scene state is preserved?

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
