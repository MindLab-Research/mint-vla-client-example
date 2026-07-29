# MANO数据处理链、资产与Provenance

本文是formal client中关于MANO数据的canonical位置总览。它区分四种容易混淆的东西：原始轨迹、运动学图像渲染、训练时动态投影、真实物理replay。路径状态核对日期为2026-07-29。

## 1. 总图

```text
外部生成的MANO NPY轨迹 + source identity/language metadata
        │
        ├──> 原始Dataset_B Lance（7,539 rows，无图像、无urdf_dof_target）
        │        │
        │        ├── 运动学相机渲染：逐帧强写hand/object qpos + mj_forward
        │        │      └── image / wrist_image
        │        │
        │        └── 基础训练列：state / measured-delta actions / generic prompt
        │
        └──> image-enriched Lance v17
                 │
                 ├── fail-closed迁移 recorded urdf_dof_target
                 │      └── canonical Lance v20
                 │
                 ├── gesture index sidecar ──> runtime gesture prompt
                 ├── contact-window sidecar ──> training/eval frame population
                 │
                 ├── runtime训练投影
                 │      ├── B-schema absolute target → query-relative xyz/finger
                 │      ├── 32D state：qpos + five-finger contact + lift
                 │      ├── population-specific norm
                 │      └── StateAug（只存在于batch构造期间）
                 │
                 └── recorded-target MuJoCo physics replay
                        └── per-row trace/grade + client statistics snapshot
```

用户概括的三个分支是主体，但还必须包含：基础训练列、gesture index、contact-window、B动作投影、32D state、normalization、StateAug和provenance/quality sidecar。这些决定模型真正看到的样本，不是外围元数据。

## 2. 数据与sidecar的权威位置

| 角色 | 权威路径 | 当前状态 |
|---|---|---|
| 原始共享Dataset_B | `/vePFS-Mindverse/share/ylang/datasets/Dataset_B/new_all_generated_mano.lance` | Lance version236；7,539 rows；约4.9GiB；字段为`index/trajectory_metadata/timestamp/hands/objects/contact`；无图像、无target |
| 正式训练/推理Lance | `/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance` | Lance version20；7,539 rows、6,866,627 frames；约809GiB；包含head/wrist JPEG、基础训练列和`hands[].urdf_dof_target` |
| pre-target rollback | 上述Lance的tag `pre_mano_target` | version17；迁移target前的可回滚版本 |
| gesture/language index | `config/datasets/new_all_generated_mano.index.json` | 7,539 entries；17 objects、15 gestures、91 object-gesture strata；SHA256 `ec847b5dc3fa5f59e03849bec71e1eb5d2d8557ad0addfa2b52feba15ba0580f` |
| contact-window manifest | `/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.contact_ctx100_error_v1.json` | target-object contact前后各100帧；SHA256 `49b6f843dc8a56132f915b145d5b2edb8d31bd8568c5b652d8e21ffa948b3731` |
| target-DOF physics evidence | `/vePFS-Mindverse/user/intern/wenxi/results/physics_quality/mano_target_physics_200hz_v1_20260725` | 17 objects、7,539/7,539 rows、0 invalid；逐行JSON/NPZ和per-object aggregate/Lance sidecar |
| client replay统计快照 | `results/training/mano_target_dof_physics_replay_stats_20260729/` | 可复算`summary.json`、CSV、README和artifact hashes；不复制raw traces |
| 当前cube1+cube2 norm | `results/training/cube1_cube2_all_32d_extended_norm_v1_20260728/` | rows507–2503；SHA256 `4f91eca8ee91d53426ea07faf28873ab98c3761ecb84d6374f4c0c439d51069a` |

`results/`在formal client中有意Git-ignored。脚本、契约和文档进Git；大型数据、run输出和checkpoint留在PFS。

## 3. 语言标注

语言有两个层次，不能把它们当成同一列：

1. Canonical Lance中的`prompt`由外部脚本生成，形式为`pick up the <object>`，只有物体任务。
2. `config/datasets/new_all_generated_mano.index.json`保存semantic gesture label。每行通过`row_index + uuid + seed_uuid + object_type + total_frames`与Lance fail-closed对齐，`gesture/action_id`是两位动作类别。训练时客户端把prompt变成：

```text
pick up the <object> using gesture <XX>
```

客户端实现：

- `scripts/gesture_language.py::GestureIndex`：校验index版本、行序、UUID、source identity和gesture一致性。
- `scripts/gesture_language.py::format_gesture_prompt`：构造runtime prompt。
- `scripts/train/train_cube1_01_compare.py`：在datum构造时读取并注入gesture prompt。

当前index自身是完整可用的标签源，但生成它的上游脚本没有进入formal client。JSON记录的原始NPY root `/mnt/nas-222-project/.../new_all_with_keypoints`和source Lance index `/home/jay/dexrobot/.../lance_human_p1_remake_combined_index.json`在当前服务器均未挂载。这是尚未闭合的provenance缺口。

## 4. 运动学图像Dataset生成

生产代码在独立、已tracked的`pi-finetune`仓库：

```text
/vePFS-Mindverse/user/intern/wenxi/pi-finetune
commit e18bb9eca6718e56cec2ec363f273a1cc72cb43c
```

关键文件：

- `case/01_export_video/render_dataset_b_images.py`
- `case/01_export_video/export_mano_sim_video.py`
- `case/01_export_video/add_training_fields_dataset_b.py`
- `case/01_export_video/configs/pi_video_streams.json`
- `case/01_export_video/schema/pi0_mujoco_training_schema.jsonc`

准确语义是“使用MuJoCo模型和renderer做运动学重放”，不是“不使用MuJoCo”。每个frame都从Lance读取并强写：

- `hands[0].urdf_dof[t]`
- `objects[0].pos[t]`
- `objects[0].rot_aa[t]`

随后只调用`mj_forward`和head/wrist camera render，不调用`mj_step`，不让物体由动力学演化。因此生成的JPEG是reference pose的视觉投影，不是物理可行性证明。

生产过程分16个shard并按原始row顺序合并。日志：

```text
/vePFS-Mindverse/user/intern/wenxi/results/logs/dataset_b_render/
```

`add_training_fields_dataset_b.py`随后加入：

```text
state[t]   = pad32(urdf_dof[t])
actions[t] = pad32(urdf_dof[t+1] - urdf_dof[t])
prompt     = "pick up the <object>"
episode_metadata
```

这里持久化的`actions`是历史M-schema measured delta，不是当前B-schema训练标签。B标签在客户端runtime从`urdf_dof_target`重新投影，不能直接训练Lance里的`actions`列来声称是B。

## 5. recorded target DOF的迁移与训练投影

Canonical v20中的目标字段为：

```text
hands[0].urdf_dof_target: [T, 26]
```

迁移代码在formal client：

```text
scripts/tools/migrate_mano_target_column.py
```

它要求target source与image Lance逐行完全一致，只允许在`hands`中新增`urdf_dof_target`；迁移使用Lance中间版本、原子rename/drop和`pre_mano_target` rollback tag。生产证据：

```text
/vePFS-Mindverse/user/intern/wenxi/results/logs/dataset_b_target_migration/
```

原staging input曾位于：

```text
/vePFS-Mindverse/user/intern/wenxi/results/datas/staging/new_all_generated_mano_with_target.lance
```

该目录目前已被清理。v20 target、v17 rollback和迁移报告仍完整，但“从原始生成源重新生产target staging dataset”的代码/输入没有在formal client闭合。

当前正式B-schema代码：

- `scripts/target_actions.py`
- `scripts/train/openpi_vla_smoke_lance_base.py`
- `scripts/train/train_cube1_01_compare.py`

布局：

```text
action = [xyz3 | Euler3 | finger20 | padding6]
horizon = 10
DeltaActions mask = 11100011111111111111111111000000
```

`urdf_dof_target`在Lance中保持absolute。OpenPI只把xyz和20个finger DOF变成相对当前query state的residual；Euler保持absolute；padding保持physical zero。

## 6. Contact window、32D state、norm与StateAug

这些不是另一个Lance copy，而是客户端消费canonical Lance时的动态处理。

### Contact window

- `scripts/contact_windows.py`：定义target-object contact record presence和窗口契约。
- `scripts/tools/build_contact_windows.py`：生成/扩展manifest。
- 当前canonical manifest：contact区间前后各100 source frames；不存在contact时必须显式选择full/skip/error策略。

### 32D state

- `scripts/mano_state_contract.py`
- `scripts/train/train_cube1_01_compare.py`

```text
state[0:26]  = hand qpos
state[26:31] = index/thumb/ring/middle/pinky contact
state[31]    = object_z[t] - object_z[0]
```

训练contact来自Lance target-object contact-record presence；lift从`objects[].pos[:,2]`动态计算。Canonical Lance持久化的旧`state[26:32]`是零，不能把它当作当前32D v1特征。

### Population-specific norm

Norm必须针对精确的rows、frame window、state contract和action semantics重新计算并通过SHA256 fail-closed认证。当前cube1+cube2 norm的可复算脚本在：

```text
results/training/cube1_cube2_all_32d_extended_norm_v1_20260728/compute_norm.py
```

### StateAug

StateAug只存在于训练batch构造期间：quantile normalization之后、`TokenizePrompt`之前，对normalized `state[0:26]`加Gaussian noise，并同步调整B动作的xyz/finger residual以保持absolute PD target不变。它不修改Lance、JPEG、MuJoCo初态、contact/lift或推理输入。

## 7. recorded-target真实物理replay

物理replay从canonical v20读取recorded `urdf_dof_target`，只在frame0初始化hand/object；之后物体由MuJoCo拥有。契约为：

```text
source interval = 0.005 s (200Hz)
MuJoCo dt        = 0.0025 s
steps/interval   = 2
target offset    = 0
```

物理场景包含gravity、collision、inertia、friction和26D position servo。它与运动学图像渲染的本质差异是：运动学分支每帧强写object reference pose；物理分支只在起点写一次，之后执行`mj_step`。

生产代码冻结在结果root：

```text
/vePFS-Mindverse/user/intern/wenxi/results/physics_quality/
mano_target_physics_200hz_v1_20260725/code/
```

其中`mano_physics_quality.py`生成逐行JSON/NPZ和per-object aggregate；`render_physics_quality_video.py`生成验证过trace一致性的physics-vs-reference视频。formal client当前维护相同已验证场景/servo机制的核心在：

```text
scripts/eval/mano_physics_core.py
```

但全量quality-dataset生成器尚未正式迁入client source。客户端只有统计快照和Mode4共享physics core。

## 8. 资产

| 资产 | 位置 | 使用者 |
|---|---|---|
| MANO hand URDF | `/vePFS-Mindverse/user/intern/wenxi/pi-finetune/3rd-party/all_assets/Assets/HAND/s02/mano/Z_upNew/mano_hand.urdf` | 运动学渲染、physics replay、Mode4 |
| MANO hand meshes | 同目录下`meshes/` | visual/collision geometry |
| object URDF | `/vePFS-Mindverse/share/ylang/all_assets/Assets/sim/mano_objects_urdf/` | 17个训练对象及其他场景对象 |
| object meshes | `/vePFS-Mindverse/share/ylang/all_assets/Assets/sim/mano_assets/objects/` | object URDF引用的visual/collision mesh |
| head/wrist cameras | `render_dataset_b_images.py`和`scripts/eval/mano_action_support.py`中的显式常量 | image generation、Mode3、Mode4 |
| physics controller/contacts | `scripts/eval/mano_physics_core.py` | target replay、Mode4 |

Hand URDF当前SHA256为`136d32b72ba811f20f7ba3162af5bf7fc00e7c845edcd80b535871492daf5f87`。现有replay manifest记录dataset/index/code identity，但没有对整个hand/object asset bundle和camera constants做统一content-addressed manifest。

## 9. 当前仍缺的东西

### P0：阻碍从零重建

1. **原始synthetic NPY → raw Lance/index的producer没有归档到formal client。** Gesture index记录了source path、source SHA和匹配方法，但原root当前未挂载，生成脚本也不在client。
2. **recorded target staging的生成源已清理。** Canonical v20可直接训练和replay，但从上游重新生成`urdf_dof_target`的路径未闭合。
3. **全量physics-quality generator仍在结果快照中。** 它应迁入client并直接复用`mano_physics_core.py`，移除对retired checkout的依赖。

### P1：阻碍严格可复现/自动筛选

4. **缺少统一asset/camera manifest。** 应记录hand URDF、所有引用mesh、17个object URDF/mesh、相机常量和physics parameters的SHA256。
5. **缺少一个端到端dataset release manifest。** 目前provenance分散在gesture index、render logs、Lance version/tag、migration logs、contact manifest和norm目录中。
6. **Replay grade尚未作为canonical全局sidecar接入sampler/evaluator。** 已有17个per-object shards和客户端统计，但还没有一个row-aligned全局A/B/C sidecar供训练过滤或评估自动分层。

### 有意动态生成，不应误认为缺列

7. Gesture-expanded prompt、contact/lift 32D state、B-schema actions、StateAug和population norm本来就应在客户端runtime生成。把它们固化回809GiB Lance会增加多份易漂移的数据真相；正确做法是锁定代码、sidecar、population和SHA。

## 10. 当前责任边界

- `pi-finetune @ e18bb9e...`拥有相机/场景构建和原始image-Lance生产代码。
- Formal client拥有target迁移、gesture消费、contact windows、state/action投影、norm、StateAug、training和Mode4。
- PFS `results/datas`拥有canonical大型数据；`results/physics_quality`拥有raw physics-quality evidence。
- MINT/OpenPI不拥有数据定义；它们消费客户端发送的已投影、已归一化batch。

任何新数据population至少要一起锁定：Lance path/version、ordered row population、gesture-index SHA、contact-manifest SHA、state/action contract、norm SHA、asset/scene identity和source commit。
