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
