# `020_hammer` asset contract

- Logical asset: `assets/objects/020_hammer`, model ID `0`.
- Metadata scale: `[0.079, 0.079, 0.079]`; `create_actor()` 自动采用该值。
- `functional point 0` 表示 hammer head，用于 BeatBlockHammer 的目标对齐与成功判定。
- `contact point 0` 位于 handle，供专家 grasp 逻辑使用。
- Metadata 标记 `stable=false`；官方任务通过指定 pose、低 mass 和场景稳定性检查使用它。

本卡片只在修改或替换 hammer、functional/contact point 或 asset 外观时检索。单纯修改 block 颜色时不要注入。
