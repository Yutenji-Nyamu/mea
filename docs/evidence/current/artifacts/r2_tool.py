{
  "schema_version": 1,
  "status": "passed",
  "route": "bound_llm_generated_checker",
  "tool_spec": {
    "task_name": "click_bell",
    "metric": "click_target_without_distractor_success"
  },
  "episodes": [
    {
      "episode_dir": "/root/autodl-tmp/mea/mea/generated_tasks/run_20260727_batch26_clean_online_click_live_v6_round_2/evaluation/telemetry/act/episode_000_seed_100405",
      "policy_name": "ACT",
      "role": "policy_under_evaluation",
      "seed": 100405,
      "metadata": {
        "schema_version": 1,
        "recorder_schema_version": 2,
        "task_name": "click_bell",
        "task_module": "mea.generated_tasks.run_20260727_batch26_clean_online_click_live_v6_round_2.task",
        "task_config": "demo_clean",
        "checkpoint_setting": "demo_clean",
        "policy_name": "ACT",
        "seed": 100405,
        "episode_index": 0,
        "success": true,
        "policy_steps": 63,
        "physics_steps": 4816,
        "physics_timestep_seconds": 0.004,
        "simulation_duration_seconds": 19.264,
        "wall_duration_seconds": 21.217623233795166,
        "policy_state_rows": 65,
        "semantic_trace_rows": 4817,
        "dynamics_trace_rows": 965,
        "telemetry_profile_id": "balanced_v1",
        "telemetry_profile_sha256": "f13e1b86e74d1f203bd9a889191203b4a1ff87d339e557aac864c027f715024c",
        "telemetry": {
          "profile_id": "balanced_v1",
          "profile_sha256": "f13e1b86e74d1f203bd9a889191203b4a1ff87d339e557aac864c027f715024c",
          "profile_artifact": "telemetry_profile.json",
          "streams": {
            "policy_state": {
              "artifact": "states.csv",
              "sampling": "policy_boundary",
              "rows": 65
            },
            "semantic_trace": {
              "artifact": "semantic_trace.npz",
              "sampling": "physics_period",
              "every_physics_steps": 1,
              "rows": 4817,
              "arrays": {
                "bell_contact_position": {
                  "shape": [
                    4817,
                    3
                  ],
                  "dtype": "float32"
                },
                "bell_position": {
                  "shape": [
                    4817,
                    3
                  ],
                  "dtype": "float32"
                },
                "left_tcp_position": {
                  "shape": [
                    4817,
                    3
                  ],
                  "dtype": "float32"
                },
                "physics_step": {
                  "shape": [
                    4817
                  ],
                  "dtype": "float64"
                },
                "policy_step": {
                  "shape": [
                    4817
                  ],
                  "dtype": "float64"
                },
                "right_tcp_position": {
                  "shape": [
                    4817,
                    3
                  ],
                  "dtype": "float32"
                },
                "simulation_time_seconds": {
                  "shape": [
                    4817
                  ],
                  "dtype": "float64"
                },
                "success": {
                  "shape": [
                    4817
                  ],
                  "dtype": "bool"
                }
              }
            },
            "contact_events": {
              "artifact": "events.jsonl",
              "sampling": "physics_period",
              "every_physics_steps": 1,
              "mode": "interval_summary",
              "rows": 2
            },
            "dynamics_trace": {
              "artifact": "dynamics_trace.npz",
              "sampling": "physics_period",
              "every_physics_steps": 5,
              "force_initial_sample": true,
              "force_final_sample": true,
              "rows": 965,
              "arrays": {
                "actor.bell.angular_velocity": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.bell.contact.0.position": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.bell.contact.0.quaternion": {
                  "shape": [
                    965,
                    4
                  ],
                  "dtype": "float32"
                },
                "actor.bell.linear_velocity": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.bell.position": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.bell.quaternion": {
                  "shape": [
                    965,
                    4
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.angular_velocity": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.contact.0.position": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.contact.0.quaternion": {
                  "shape": [
                    965,
                    4
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.linear_velocity": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.position": {
                  "shape": [
                    965,
                    3
                  ],
                  "dtype": "float32"
                },
                "actor.distractor.quaternion": {
                  "shape": [
                    965,
                    4
                  ],
                  "dtype": "float32"
                },
                "physics_step": {
                  "shape": [
                    965
                  ],
                  "dtype": "int64"
                },
                "policy_step": {
                  "shape": [
                    965
                  ],
                  "dtype": "int64"
                },
                "robot.left.ee_pose": {
                  "shape": [
                    965,
                    7
                  ],
                  "dtype": "float32"
                },
                "robot.left.gripper": {
                  "shape": [
                    965
                  ],
                  "dtype": "float32"
                },
                "robot.left.qpos": {
                  "shape": [
                    965,
                    38
                  ],
                  "dtype": "float32"
                },
                "robot.left.qvel": {
                  "shape": [
                    965,
                    38
                  ],
                  "dtype": "float32"
                },
                "robot.left.tcp_pose": {
                  "shape": [
                    965,
                    7
                  ],
                  "dtype": "float32"
                },
                "robot.right.ee_pose": {
                  "shape": [
                    965,
                    7
                  ],
                  "dtype": "float32"
                },
                "robot.right.gripper": {
                  "shape": [
                    965
                  ],
                  "dtype": "float32"
                },
                "robot.right.qpos": {
                  "shape": [
                    965,
                    38
                  ],
                  "dtype": "float32"
                },
                "robot.right.qvel": {
                  "shape": [
                    965,
                    38
                  ],
                  "dtype": "float32"
                },
                "robot.right.tcp_pose": {
                  "shape": [
                    965,
                    7
                  ],
                  "dtype": "float32"
                },
                "simulation_time_seconds": {
                  "shape": [
                    965
                  ],
                  "dtype": "float64"
                },
                "success": {
                  "shape": [
                    965
                  ],
                  "dtype": "bool"
                }
              }
            }
          }
        },
        "contact_interval_count": 1,
        "error": null,
        "artifacts": {
          "policy_states": "states.csv",
          "semantic_trace": "semantic_trace.npz",
          "dynamics_trace": "dynamics_trace.npz",
          "events": "events.jsonl",
          "task_schema": "schema.json",
          "telemetry_profile": "telemetry_profile.json",
          "video": "video.mp4"
        },
        "video_alignment": {
          "policy_frame_rate_hz": 10,
          "frame_semantics": "pre-action; contact in policy step k lies between adjacent frames"
        }
      },
      "result": {
        "tool": "click_target_without_distractor_success",
        "value": true,
        "unit": null,
        "passed": true,
        "evidence_steps": [
          4816
        ],
        "details": {
          "authority": "llm_generated_python_ast_validated",
          "official_success": false,
          "proposal_sha256": "2cdada0c0b7004c73ad5cc5633fa9fb2b00531d5eb656ace189ac04c6469f28e",
          "module_sha256": "ff3e225dc5eb1e46ce2b5cf1a736cbf4148de222c5b35d0741513d40e6a4fbde",
          "success_method_sha256": "1dcfaae0a1858ee2ab51df0df05cc2d16a22b53f8766059e794536d77f35ff51",
          "task_module": "mea.generated_tasks.run_20260727_batch26_clean_online_click_live_v6_round_2.task",
          "generated_checker_success": true,
          "official_core_predicate_satisfied": true,
          "distractor_contact_latched": false,
          "distractor_latch_authority": "logical_implication_of_validated_checker_success",
          "distractor_contact_event_recorded": false,
          "distractor_trace_coverage": "not_registered_in_current_click_bell_task_schema"
        }
      }
    }
  ]
}
