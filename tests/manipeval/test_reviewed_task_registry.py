import hashlib
import json
import os
import tempfile
import textwrap
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mea.capability_adapter import resolve_capability_contract
from mea.proposals import task_proposal_from_contract
from mea.taskgen.artifacts import write_task_artifact_bundle
from mea.taskgen.capabilities import build_variant_spec
from mea.taskgen.prototype import validate_load_actors
from mea.taskgen.prototype import compile_overlay
from mea.taskgen.resolver import task_semantic_key
from mea.taskgen.reviewed_registry import (
    RUNTIME_DEPENDENCY_PATHS,
    ReviewedTaskRegistryError,
    build_task_review_manifest_template,
    copy_reviewed_task_artifacts,
    find_reviewed_task,
    install_reviewed_task,
    load_reviewed_task_registry,
)
from mea.taskgen.success_spec import (
    compile_success_spec,
    default_bbh_success_spec,
    experimental_bbh_success_spec_v2,
)


LOAD_ACTORS = """
def load_actors(self):
    self.hammer = create_actor(
        scene=self,
        pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
        modelname="020_hammer",
        convex=True,
        model_id=0,
    )
    block_pose = rand_pose(
        xlim=[-0.25, 0.25],
        ylim=[-0.05, 0.15],
        zlim=[0.76],
        qpos=[1, 0, 0, 0],
        rotate_rand=True,
        rotate_lim=[0, 0, 0.5],
    )
    while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
    self.block = create_box(
        scene=self,
        pose=block_pose,
        half_size=(0.025, 0.025, 0.025),
        color=(0.0, 0.2, 1.0),
        name="box",
        is_static=True,
    )
    self.hammer.set_mass(0.001)
    self.add_prohibit_area(self.hammer, padding=0.10)
    self.prohibited_area.append([
        block_pose.p[0] - 0.05,
        block_pose.p[1] - 0.05,
        block_pose.p[0] + 0.05,
        block_pose.p[1] + 0.05,
    ])
"""


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class ReviewedTaskRegistryTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.source_run = self.root / "mea/generated_tasks/source_run"
        self.registry = self.root / "reviewed_tasks"
        for relative in RUNTIME_DEPENDENCY_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# pinned test dependency: {relative}\n", encoding="utf-8")
        contract = resolve_capability_contract(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        proposal = task_proposal_from_contract(
            contract, intent="evaluate a blue block"
        )
        self.semantic_key = task_semantic_key(proposal, contract)
        self.query = {
            "schema_version": 1,
            "semantic_key": self.semantic_key,
            "semantic_key_sha256": canonical_sha256(self.semantic_key),
        }
        self._write_source_run()

    def tearDown(self):
        self._temporary.cleanup()

    def _write_source_run(self):
        variant = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_appearance.color_blue",
            capability_id="object_appearance.color",
            intent="planner capability: blue block",
            changes=self.semantic_key["changes"],
            generation_mode="force_codegen",
        )
        success_spec = default_bbh_success_spec()
        success_method, success_validation = compile_success_spec(success_spec)
        source = (
            '"""Reviewed TaskGen fixture."""\n\n'
            "import numpy as np\n"
            "import sapien\n\n"
            "from envs.beat_block_hammer import beat_block_hammer as OfficialBeatBlockHammer\n"
            "from envs.utils import create_actor, create_box, rand_pose\n\n\n"
            "class beat_block_hammer(OfficialBeatBlockHammer):\n"
            + textwrap.indent(textwrap.dedent(LOAD_ACTORS).strip(), "    ")
            + "\n\n"
            + textwrap.indent(success_method.strip(), "    ")
            + "\n"
        )
        self.source_run.mkdir(parents=True)
        (self.source_run / "task.py").write_text(source, encoding="utf-8")
        write_json(self.source_run / "variant_spec.json", variant)
        (self.source_run / "overlay.yml").write_text(
            yaml.safe_dump(
                compile_overlay(variant), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        (self.source_run / "generation").mkdir(parents=True, exist_ok=True)
        (self.source_run / "generation/load_actors.py.txt").write_text(
            textwrap.dedent(LOAD_ACTORS).strip() + "\n", encoding="utf-8"
        )
        write_json(
            self.source_run / "generation/success_spec.json", success_spec
        )
        bundle = write_task_artifact_bundle(
            self.root,
            self.source_run,
            {
                "task_name": "beat_block_hammer",
                "task_module": "mea.generated_tasks.source_run.task",
                "mode": "force_codegen",
                "generation_kind": "generated_scene_code",
            },
        )
        self.assertEqual(bundle["scene_method"]["origin"], "generated_code")
        write_json(
            self.source_run / "validation/static.json",
            {
                "variant_spec": {"valid": True},
                "success_spec": success_validation,
                "load_actors_ast": validate_load_actors(LOAD_ACTORS, variant),
                "protected_diff": {"valid": True, "hashes_after": {}},
            },
        )

    def _approved_manifest(self):
        manifest = build_task_review_manifest_template(
            self.source_run, self.semantic_key
        )
        manifest.update(
            {
                "decision": "approved",
                "reviewer": {
                    "id": "unit-test-development-review",
                    "kind": "development_agent",
                },
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "notes": "Test-only review of every pinned artifact.",
            }
        )
        manifest["checks"] = {key: True for key in manifest["checks"]}
        return manifest

    def _install(self):
        return install_reviewed_task(
            self.source_run,
            self.semantic_key,
            self._approved_manifest(),
            self.registry,
        )

    def test_pending_or_hash_mismatched_review_never_installs(self):
        pending = build_task_review_manifest_template(
            self.source_run, self.semantic_key
        )
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "decision must be approved"
        ):
            install_reviewed_task(
                self.source_run,
                self.semantic_key,
                pending,
                self.registry,
            )
        self.assertFalse(self.registry.exists())

        approved = self._approved_manifest()
        approved["task_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "does not match source artifacts"
        ):
            install_reviewed_task(
                self.source_run,
                self.semantic_key,
                approved,
                self.registry,
            )
        self.assertFalse(self.registry.exists())

    def test_install_find_and_copy_exact_reviewed_artifacts(self):
        installed = self._install()
        index = load_reviewed_task_registry(self.registry)
        self.assertEqual(index["scope"], "reviewed_generated_task_reuse")
        self.assertEqual(len(index["entries"]), 1)
        found = find_reviewed_task(self.registry, self.query)
        self.assertIsNotNone(found)
        self.assertEqual(found["registration_id"], installed["registration_id"])
        self.assertEqual(found["semantic_key"], self.semantic_key)
        self.assertEqual(
            found["review_authority"]["kind"], "development_agent"
        )
        self.assertFalse(found["review_attestation_paper_eligible"])
        self.assertEqual(
            set(found["verified_artifacts"]),
            {
                "task.py",
                "variant_spec.json",
                "overlay.yml",
                "generation/load_actors.py.txt",
                "generation/success_spec.json",
                "generation/task_artifact_bundle.json",
                "generation/scene_check_spec.json",
                "validation/static.json",
            },
        )

        destination = self.root / "materialized"
        copied = copy_reviewed_task_artifacts(found, destination)
        self.assertEqual(set(copied["files"]), set(found["verified_artifacts"]))
        for relative, digest in copied["files"].items():
            self.assertEqual(
                hashlib.sha256((destination / relative).read_bytes()).hexdigest(),
                digest,
            )
        with self.assertRaisesRegex(ReviewedTaskRegistryError, "must be empty"):
            copy_reviewed_task_artifacts(found, destination)

    def test_runtime_dependency_drift_and_match_provenance_fail_closed(self):
        match = self._install()
        self.assertIsNotNone(
            find_reviewed_task(self.registry, self.query, repo_root=self.root)
        )
        tampered_match = json.loads(json.dumps(match, default=str))
        tampered_match["review_authority"] = {
            "id": "fake-human",
            "kind": "human",
        }
        tampered_match["review_attestation_paper_eligible"] = True
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "review_authority differs"
        ):
            copy_reviewed_task_artifacts(
                tampered_match, self.root / "tampered_copy", repo_root=self.root
            )

        dependency = self.root / RUNTIME_DEPENDENCY_PATHS[0]
        dependency.write_text("# drifted runtime dependency\n", encoding="utf-8")
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "runtime dependencies changed"
        ):
            find_reviewed_task(self.registry, self.query, repo_root=self.root)

    def test_same_artifact_different_review_attestation_is_not_silent(self):
        self._install()
        changed_review = self._approved_manifest()
        changed_review["reviewer"] = {
            "id": "different-development-review",
            "kind": "development_agent",
        }
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "different review attestation"
        ):
            install_reviewed_task(
                self.source_run,
                self.semantic_key,
                changed_review,
                self.registry,
            )

    def test_preserved_semantic_key_rejects_experimental_success_source(self):
        success_spec = experimental_bbh_success_spec_v2()
        success_method, success_validation = compile_success_spec(success_spec)
        task_source = (
            '"""Reviewed TaskGen fixture."""\n\n'
            "import numpy as np\n"
            "import sapien\n\n"
            "from envs.beat_block_hammer import beat_block_hammer as OfficialBeatBlockHammer\n"
            "from envs.utils import create_actor, create_box, rand_pose\n\n\n"
            "class beat_block_hammer(OfficialBeatBlockHammer):\n"
            + textwrap.indent(textwrap.dedent(LOAD_ACTORS).strip(), "    ")
            + "\n\n"
            + textwrap.indent(success_method.strip(), "    ")
            + "\n"
        )
        (self.source_run / "task.py").write_text(task_source, encoding="utf-8")
        write_json(
            self.source_run / "generation/success_spec.json", success_spec
        )
        proposal_v2 = {
            "schema_version": 2,
            "proposal_id": "object_appearance.experimental_success",
            "task_name": "beat_block_hammer",
            "aspect_id": "object_appearance.color",
            "intent": "bounded experimental threshold",
            "capability_id": "object_appearance.color",
            "reuse_first": True,
            "changes": self.semantic_key["changes"],
            "preserve_success_semantics": False,
            "success_spec": success_spec,
        }
        write_json(
            self.source_run / "generation/task_proposal.json", proposal_v2
        )
        write_task_artifact_bundle(
            self.root,
            self.source_run,
            {
                "task_name": "beat_block_hammer",
                "task_module": "mea.generated_tasks.source_run.task",
                "mode": "force_codegen",
                "generation_kind": "generated_scene_code",
            },
            task_proposal=proposal_v2,
        )
        static_path = self.source_run / "validation/static.json"
        static = json.loads(static_path.read_text(encoding="utf-8"))
        static["success_spec"] = success_validation
        write_json(static_path, static)

        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "preservation claim differs"
        ):
            build_task_review_manifest_template(
                self.source_run, self.semantic_key
            )

    def test_non_exact_semantic_key_does_not_match(self):
        changed = json.loads(json.dumps(self.semantic_key))
        changed["changes"]["block"]["color"] = [0.0, 1.0, 0.0]
        query = {
            "schema_version": 1,
            "semantic_key": changed,
            "semantic_key_sha256": canonical_sha256(changed),
        }
        self._install()
        self.assertIsNone(find_reviewed_task(self.registry, query))

    def test_overlay_and_repair_source_must_match_task_contract(self):
        overlay_path = self.source_run / "overlay.yml"
        original_overlay = overlay_path.read_bytes()
        overlay_path.write_text("mea:\n  enabled: false\n", encoding="utf-8")
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "overlay.yml does not match"
        ):
            build_task_review_manifest_template(self.source_run, self.semantic_key)
        overlay_path.write_bytes(original_overlay)

        repair_path = self.source_run / "generation/load_actors.py.txt"
        repair_path.write_text(
            repair_path.read_text(encoding="utf-8") + "\n# drifted repair input\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "differs from task.py"
        ):
            build_task_review_manifest_template(self.source_run, self.semantic_key)

    def test_tampered_artifact_fails_load_and_copy(self):
        match = self._install()
        task_path = Path(match["verified_artifacts"]["task.py"]["path"])
        task_path.write_text(
            task_path.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ReviewedTaskRegistryError, "tampered"):
            load_reviewed_task_registry(self.registry)
        with self.assertRaisesRegex(ReviewedTaskRegistryError, "tampered"):
            copy_reviewed_task_artifacts(match, self.root / "copy_after_tamper")

    def test_index_path_escape_is_rejected(self):
        self._install()
        index_path = self.registry / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["entries"][0]["artifacts"]["task.py"]["path"] = (
            "../../outside/task.py"
        )
        write_json(index_path, index)
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "descriptor is invalid"
        ):
            load_reviewed_task_registry(self.registry)

    def test_unindexed_or_nonempty_storage_is_rejected(self):
        self.registry.mkdir()
        (self.registry / "surprise.txt").write_text("not indexed", encoding="utf-8")
        with self.assertRaisesRegex(ReviewedTaskRegistryError, "no index"):
            load_reviewed_task_registry(self.registry)

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows")
    def test_source_and_registry_symlinks_are_rejected(self):
        external = self.root / "external_task.py"
        external.write_text(
            (self.source_run / "task.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (self.source_run / "task.py").unlink()
        (self.source_run / "task.py").symlink_to(external)
        with self.assertRaisesRegex(ReviewedTaskRegistryError, "symlinks"):
            build_task_review_manifest_template(self.source_run, self.semantic_key)


if __name__ == "__main__":
    unittest.main()
