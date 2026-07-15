import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mea.knowledge import (
    build_knowledge_index,
    build_knowledge_index_data,
    source_symbol_text,
)
from mea.retrieval import KnowledgeRetriever


VARIANT_SPEC = {
    "task_name": "beat_block_hammer",
    "changes": {
        "block": {
            "position_mode": "official_random",
            "yaw_mode": "official_random",
            "scale": 1.0,
            "color": [0.0, 0.2, 1.0],
        }
    },
}


class KnowledgeRetrievalTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        source_root = Path(__file__).resolve().parents[2]

        for relative in (
            "mea/knowledge/tasks/beat_block_hammer.md",
            "mea/knowledge/api/scene_creation.md",
            "mea/knowledge/assets/020_hammer.md",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                (source_root / relative).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        self._write(
            "envs/beat_block_hammer.py",
            """\
class beat_block_hammer:
    def load_actors(self):
        self.block = create_box(color=(1.0, 0.0, 0.0))

    def check_success(self):
        return self.block is not None
""",
        )
        self._write(
            "envs/blocks_ranking_rgb.py",
            """\
class blocks_ranking_rgb:
    def load_actors(self):
        self.blocks = [create_box(color=(0.0, 0.2, 1.0))]
""",
        )
        self._write(
            "envs/utils/create_actor.py",
            """\
def create_box(scene, pose, half_size, color, name, is_static):
    return scene.create_box(pose, half_size, color, name, is_static)

def create_actor(scene, pose, modelname, convex, model_id):
    return scene.create_actor(pose, modelname, convex, model_id)
""",
        )
        self._write(
            "envs/utils/rand_create_actor.py",
            """\
def rand_pose(xlim, ylim, zlim, qpos, rotate_rand, rotate_lim):
    return (xlim, ylim, zlim, qpos, rotate_rand, rotate_lim)
""",
        )
        self._write(
            "assets/objects/020_hammer/model_data0.json",
            json.dumps({"scale": 1.0, "functional_matrix": []}),
        )
        self._write(
            "description/objects_description/020_hammer/base0.json",
            json.dumps({"description": "hammer"}),
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _write(self, relative, content):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def test_offline_extractor_records_symbol_and_file_hashes(self):
        index = build_knowledge_index_data(self.root)
        documents = {item["id"]: item for item in index["documents"]}

        task = documents["task.beat_block_hammer"]
        self.assertEqual(len(task["source_symbols"]), 2)
        for source in task["source_symbols"]:
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(source["character_count"], 0)
            extracted = source_symbol_text(
                self.root, source["path"], source["symbol"]
            )
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
            )

        hammer = documents["asset.020_hammer"]
        self.assertEqual(len(hammer["source_files"]), 2)
        for source in hammer["source_files"]:
            path = self.root / source["path"]
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_blue_block_retrieval_is_compact_and_excludes_asset(self):
        build_knowledge_index(self.root)
        output_dir = self.root / "run/generation"
        result = KnowledgeRetriever(self.root).select(
            "Change the red block to blue and preserve all other behavior.",
            "beat_block_hammer",
            VARIANT_SPEC,
            ["beat_block_hammer", "blocks_ranking_rgb"],
            output_dir=output_dir,
        )

        self.assertEqual(
            [item["id"] for item in result["selected_documents"]],
            ["task.beat_block_hammer", "api.scene_creation"],
        )
        self.assertNotIn("asset.020_hammer", result["selected_ids"])
        self.assertEqual(
            [item["id"] for item in result["selected_examples"]],
            ["example.blocks_ranking_rgb.load_actors"],
        )
        self.assertLessEqual(result["context_character_count"], 8000)
        self.assertEqual(
            result["context_character_count"], len(result["context"])
        )
        self.assertTrue(result["committed_index_current"])
        self.assertIn("def load_actors(self):", result["context"])
        self.assertNotIn("def check_success(self):", result["context"])

        persisted = json.loads(
            (output_dir / "knowledge_retrieval.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(persisted["selected_ids"], result["selected_ids"])
        self.assertNotIn("context", persisted)


if __name__ == "__main__":
    unittest.main()
