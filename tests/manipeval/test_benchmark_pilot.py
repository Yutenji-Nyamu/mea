import json
import tempfile
import unittest
from pathlib import Path

from mea.benchmark_pilot import BenchmarkPilotError, aggregate_three_task_pilot


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BenchmarkPilotTests(unittest.TestCase):
    def test_three_task_same_seed_aggregation_is_smoke_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = []
            for index, task_name in enumerate(("a", "b", "c"), start=1):
                direct = f"direct/{task_name}/episode.json"
                validity = f"direct/{task_name}/summary.json"
                protocol = f"mea/protocol_runs/protocol_{task_name}"
                agent_dir = f"mea/generated_tasks/run_{task_name}/evaluation/telemetry/act/episode_0"
                payload = {
                    "task_name": task_name,
                    "policy_name": "ACT",
                    "seed": index,
                    "success": True,
                    "policy_steps": 10,
                    "physics_steps": 100,
                    "simulation_duration_seconds": 1.0,
                    "wall_duration_seconds": 2.0,
                    "error": None,
                }
                write_json(root / direct, payload)
                write_json(root / validity, {"valid_for_comparison": True})
                write_json(root / f"{agent_dir}/episode.json", payload)
                write_json(
                    root / f"{protocol}/summary/protocol_summary.json",
                    {
                        "valid_for_comparison": True,
                        "agent_wall_duration_seconds": 3.0,
                    },
                )
                write_json(
                    root / f"{protocol}/protocol_manifest.json",
                    {
                        "repetitions": [
                            {
                                "attempts": [
                                    {
                                        "measurement": {
                                            "episodes": [
                                                {"seed": index, "episode_dir": agent_dir}
                                            ]
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                )
                tasks.append(
                    {
                        "task_name": task_name,
                        "seed": index,
                        "direct_episode": direct,
                        "direct_validity_artifact": validity,
                        "agent_protocol_dir": protocol,
                    }
                )
            summary = aggregate_three_task_pilot(root, {"tasks": tasks})
            self.assertEqual(summary["agreement"]["binary_same_seed_exact_count"], 3)
            self.assertEqual(summary["routes"]["mea_agent_official"]["policy_steps"], 30)
            self.assertIsNone(summary["agreement"]["table2_consistency"])
            self.assertFalse(summary["paper_table_eligible"])

            tasks[2] = dict(tasks[2], task_name="b", seed=2)
            with self.assertRaises(BenchmarkPilotError):
                aggregate_three_task_pilot(root, {"tasks": tasks})


if __name__ == "__main__":
    unittest.main()
