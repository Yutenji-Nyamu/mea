# README.Agent: RoboTwin `load_actors()`

本文档面向 TaskGen code agent，记录 RoboTwin 2.0 场景构造的稳定接口、约束与常见模式。

## Lifecycle

`Base_Task._init_task_env_()` 在调用任务的 `load_actors()` 前已经：

1. 使用 episode seed 设置 NumPy 与 Torch 随机种子。
2. 创建 SAPIEN scene、table、wall、robot 和 cameras。
3. 将 robot 移动到 home state。

`load_actors()` 返回后，框架会执行稳定性检查。继承的 `play_once()` 和 `check_success()` 会依赖任务在 `load_actors()` 中设置的 actor attributes。

## BeatBlockHammer contract

- 必须创建 `self.hammer`，模型为 `020_hammer`，`convex=True`，`model_id=0`。
- 必须创建 `self.block`，name 为 `box`。
- 官方 hammer pose 为 `sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105])`。
- 官方 block half size 为 `(0.025, 0.025, 0.025)`，是 static actor。
- 官方 block position sampling：`x ∈ [-0.25, 0.25]`、`y ∈ [-0.05, 0.15]`、`z=0.76`。
- 官方 yaw sampling 使用 `rand_pose(... rotate_rand=True, rotate_lim=[0, 0, 0.5])`。
- 必须拒绝 `abs(x) < 0.05` 或 `x²+y² < 0.001` 的采样。
- hammer mass 必须设置为 `0.001`。
- 必须为 hammer 调用 `self.add_prohibit_area(..., padding=0.10)`。
- 必须围绕 block position 向 `self.prohibited_area` 增加边长 0.10 m 的区域。
- appearance-only 变式不得覆盖 `play_once()` 或 `check_success()`。

## Stable APIs

```text
rand_pose(xlim, ylim, zlim=[0.741], ylim_prop=False,
          rotate_rand=False, rotate_lim=[0,0,0], qpos=[1,0,0,0])

create_actor(scene, pose, modelname, scale=(1,1,1), convex=False,
             is_static=False, model_id=0)

create_box(scene, pose, half_size, color=None, is_static=False,
           name="", texture_id=None, boxtype="default")

self.add_prohibit_area(actor, padding=0.01)
```

`create_box.color` 使用 `[0,1]` RGB tuple。RoboTwin table top nominal height 为约 `0.74 m`，方块中心高度必须包含 half height。

## Generation rules

- 生成完整 `def load_actors(self):`，不能调用 `super()`。
- 仅改变 VariantSpec 明确要求的变量。
- 保持官方随机数调用顺序，避免相同 seed 下出现非目标变化。
- 不导入模块；wrapper 已提供 `np`、`sapien`、`create_actor`、`create_box` 和 `rand_pose`。
- 不访问文件、网络、环境变量或进程。
- 不创建用户未要求的 actors。
