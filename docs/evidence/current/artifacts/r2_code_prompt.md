You are the TaskGen code agent for RoboTwin 2.0.

USER REQUEST:
将 beat_block_hammer 的目标方块等比例放大到官方尺寸的 1.2 倍，保持红色、官方位置/朝向采样、成功语义和其余行为不变；先通过 render 与 expert gate。 Query-generated bounded variation: evaluate a query-relevant bounded variation

VALIDATED VARIANT SPEC:
{
  "schema_version": 2,
  "task_name": "beat_block_hammer",
  "variant_id": "object_scale.query_generated_1",
  "capability_id": "object_scale.bounded",
  "intent": "evaluate a query-relevant bounded variation",
  "controlled_axis": "object_scale",
  "generation_mode": "force_codegen",
  "changes": {
    "block": {
      "color": [
        1.0,
        0.0,
        0.0
      ],
      "position_mode": "official_random",
      "scale": 1.2,
      "yaw_mode": "official_random"
    }
  },
  "preserve": [
    "official_position_sampling",
    "official_yaw_sampling",
    "official_block_color",
    "play_once",
    "check_success_semantics",
    "checkpoint"
  ]
}

Your output will be inserted into a thin subclass of the official
``envs.beat_block_hammer.beat_block_hammer`` class.

OUTPUT CONTRACT:
1. Output exactly one Python fenced code block.
2. The block must contain the complete ``def load_actors(self):`` method and nothing else.
3. Generate the complete method body yourself. Do not call ``super()``.
4. Recreate every actor, official pose sampling/rejection rule, mass setting,
   and prohibited area used by BeatBlockHammer.
5. Apply only the validated requested change. For this spec, use a literal RGB
   tuple in ``create_box(..., color=...)`` and the fully evaluated literal
   three-vector in ``create_box(..., half_size=...)``.
6. Preserve actor attribute names ``self.hammer`` and ``self.block`` because
   inherited ``play_once`` and ``check_success`` depend on them.
7. Available globals are: ``np``, ``sapien``, ``create_actor``, ``create_box``,
   ``rand_pose``, and ordinary safe builtins. Do not import anything.
8. Do not access files, network, processes, environment variables, or dynamic imports.

GLOBAL OUTPUT RULES:
# TaskGen output rules

- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
- Preserve actor identity, collision behavior, random-call order, and required
  telemetry names.
- Do not import or access files, network, environment variables, processes,
  dynamic execution, dunder attributes, or `super()`.
- Use wrapper-provided `np`, `sapien`, `create_actor`, `create_box`, and
  `rand_pose`.
- The ordinary scene-only route must not replace `check_success()`.
- The explicit `provider_scene_checker_codegen` route must generate both
  `load_actors()` and `check_success()` from the same Proposal. Its checker is
  experimental and must never be relabeled as official success.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.


OFFICIAL LOAD_ACTORS SOURCE (authoritative behavior):
```python
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
        color=(1, 0, 0),
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
```

RETRIEVED TASK/API KNOWLEDGE:
## README.Agent freshness snapshot

{
  "schema_version": 1,
  "task_name": "beat_block_hammer",
  "global_rules_path": "mea/taskgen/README.Agent.md",
  "global_rules_sha256": "2a52c838277506e5241b89ac4c3f93a5a57132a66ac8e725f87e370e72710f5d",
  "document_ids": [
    "task.beat_block_hammer",
    "api.scene_creation",
    "asset.020_hammer"
  ],
  "source_fingerprint_sha256": "20aa22ba8ad06ecc4274393ad870b3930c69c1839bd521c80daf0085549b34e0",
  "snapshot_sha256": "05e3e81897edcb3901c12377d80037c7d763d52457cd6f866c18c150e2ec7b9e"
}

## task.beat_block_hammer
Source: `mea/knowledge/tasks/beat_block_hammer.md`

# BeatBlockHammer scene contract

供 TaskGen 生成完整 `load_actors()` 时使用。官方
`envs/beat_block_hammer.py:beat_block_hammer.load_actors` 始终是行为权威；本卡片只强调容易遗漏的不变量。

- 必须创建 `self.hammer` 与 `self.block`。继承的 `play_once()` 和
  `check_success()` 依赖这两个属性及其 functional points。
- hammer 使用 `020_hammer`、`model_id=0`、`convex=True`，官方 pose 为
  `sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105])`，mass 为
  `0.001`，并调用 `self.add_prohibit_area(..., padding=0.10)`。
- block 使用 `create_box`，name 必须为 `box`、`is_static=True`，官方
  half-size 为 `(0.025, 0.025, 0.025)`。
- 官方 position sampling 为 `x∈[-0.25,0.25]`、`y∈[-0.05,0.15]`、
  `z=0.76`；拒绝 `abs(x)<0.05` 或 `x²+y²<0.001`。
- 官方 yaw 通过 `rand_pose(..., rotate_rand=True,
  rotate_lim=[0,0,0.5])` 采样。保持随机调用顺序，才能让相同 seed 的非目标变量不变。
- block 周围还要向 `self.prohibited_area` 加入左右各 `0.05 m` 的区域。
- appearance-only 变式只能改变请求指定的外观值；不得覆盖或改写
  `play_once()`、`check_success()`、actor identity、采样逻辑或碰撞设置。

## api.scene_creation
Source: `mea/knowledge/api/scene_creation.md`

# RoboTwin scene construction APIs

本卡片只覆盖 BeatBlockHammer `load_actors()` 实际需要的稳定接口。

```text
create_box(scene, pose, half_size, color=None, is_static=False,
           name="", texture_id=None, boxtype="default") -> Actor
```

- `half_size` 是三个轴的半边长，不是完整边长。
- `color` 是 `[0,1]` RGB tuple，例如蓝色 `(0.0, 0.2, 1.0)`。
- 返回 RoboTwin `Actor` wrapper；`name` 决定 scene contact 中的身份。
- `is_static=True` 创建静态刚体。

```text
create_actor(scene, pose, modelname, scale=(1,1,1), convex=False,
             is_static=False, model_id=0) -> Actor
```

- `model_id=0` 对应 `model_data0.json` 与 `base0` asset。
- 当 model metadata 存在时，函数会使用其中的 `scale`，覆盖调用方传入的
  `scale`；不要用该参数改变 `020_hammer` 的视觉尺寸。
- `convex=True` 使用分解后的 convex collision meshes；hammer 官方代码要求保留。

```text
rand_pose(xlim, ylim, zlim=[0.741], ylim_prop=False,
          rotate_rand=False, rotate_lim=[0,0,0], qpos=[1,0,0,0])
```

- quaternion 顺序为 `wxyz`。
- 即使某个范围上下界相等，函数仍会执行随机采样；`rotate_rand=True` 还会消耗姿态随机数。
- 为保持相同 seed 的场景一致，不要重排、合并或删除官方随机调用。

## example.blocks_ranking_rgb.load_actors
Source: `envs/blocks_ranking_rgb.py:blocks_ranking_rgb.load_actors`

```python
    def load_actors(self):
        while True:
            block_pose_lst = []
            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.765],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

                def check_block_pose(block_pose):
                    for j in range(len(block_pose_lst)):
                        if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                            return False
                    return True

                while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.01
                       or not check_block_pose(block_pose)):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.765],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                block_pose_lst.append(deepcopy(block_pose))
            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p
            if (np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                    and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps) and block1_pose[0] < block2_pose[0]
                    and block2_pose[0] < block3_pose[0]):
                continue
            else:
                break

        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)
        self.block1 = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1, 0, 0),
            name="box",
        )
        self.block2 = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0, 1, 0),
            name="box",
        )
        self.block3 = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)

        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        # Generate random y position for all blocks
        y_pose = np.random.uniform(-0.2, -0.1)

        # Define target poses for each block with random x positions
        self.block1_target_pose = [
            np.random.uniform(-0.09, -0.08),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(0.08, 0.09),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
```
