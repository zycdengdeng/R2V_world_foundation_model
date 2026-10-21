# R2V World Foundation Model

基于 NVIDIA Cosmos-Transfer2.5 的**多视角可控驾驶视频生成**。给定三路条件视频——布局（hdmap+bbox）、光度（blur 投影）、几何（稀疏深度）——生成 7 视角、29 帧、720p、10Hz 的驾驶视频。

![架构图](docs/assets/r2v_architecture.svg)

**方法一句话**：三路条件各自经冻结的 Wan2.1 VAE 编码为 16 通道潜变量，按"布局(0-15)/光度(16-31)/几何(32-47)"的顺序填入主干预留的 8 槽位通道接口（布局落在预训练过的 hdmap 通路上以继承先验）；主干 28 块 DiT 全程冻结，只训练 VACE 式控制分支（控制嵌入器 + hint MLP + 4 个控制块 = **329M 参数，占全模型 13.8%**），在主干第 0/7/14/21 块以零初始化的可加残差注入控制信号。训练目标为整流流速度预测，无条件帧（纯控制驱动生成）。

---

## 1. 与上游 Cosmos 的关系（改动审计）

- `cosmos_transfer2/_src/` 是上游 Cosmos 代码，**网络结构零改动**。全部修改仅 6 处 commit：
  - 条件 dataclass 增加 `control_input_blur` / `control_input_blur_mask` 两个字段（`_src/transfer2/configs/vid2vid_transfer/defaults/conditioner.py`）；
  - 5 处 batch_size>1 的张量处理修复（`_src/predict2_multiview/models/multiview_vid2vid_model_rectified_flow.py`、`view_sampling.py`），不影响 batch=1 行为。
- 我们的全部代码在 `cosmos_transfer2/experiments/custom/`、`scripts/` 和 `docs/`。
- 训练起点统一为官方检查点 `nvidia/Cosmos-Transfer2.5-2B` auto/multiview（checkpoint-db uuid `4ecc66e9-df19-4aed-9802-0d11e057287a`，带预训练 hdmap_bbox 控制头），由代码自动从 HF 镜像下载。
- 冻结策略是上游模型类的内建行为（`train_base_model=False` 时构造期自动冻结主干），非本仓库修改。

## 2. 目录地图

### 实验配置（`cosmos_transfer2/experiments/custom/`）

| 文件 | 用途 |
|---|---|
| `custom_multi_control_experiment.py` | **天津全量**训练（75 场景，8000 步；当前文件处于 warm-restart 状态，从自家 iter_5000 续训） |
| `tianjin_small_experiment.py` | **天津 small**（10 场景 / 400 步 / 测试 053+077，详见 `docs/tianjin_small_run.md`） |
| `dongfeng_multi_control_experiment.py` | **东风数据**训练（31 场景 / 8000 步，/mnt2 数据，场景自动划分） |
| `ablation_single_view_experiment.py` | 单视角控制模态消融（7 组 hint_keys 组合） |
| `dense_multi_control_experiment.py` | 稠密控制变体实验 |
| `custom_multi_control_dataset.py` | 天津数据集类（`{场景}_seg{NN}` 命名） |
| `dongfeng_multi_control_dataset.py` | 东风数据集子类（`{drive}_seg{NNN}` 命名，仅重写场景号解析） |
| `roadside_multi_control_dataset.py` | 路侧轨迹数据集 |
| `text_embedding_cache_callback.py` | 按 caption 缓存文本嵌入，跳过重复的 7B 编码器前向 |
| `test_loss_callback.py` | 每 N 步在测试集上算扩散损失（选检查点用） |
| `evaluation_callback.py` | 训练中评估回调 |
| `reinit_control_callback.py` | ⚠️ 未被任何配置引用（历史遗留，保留备查） |

### 推理脚本（同目录，按数据选择）

| 脚本 | 适用 |
|---|---|
| `inference.py` | 天津数据（全量或 small 模型，`--experiment` 区分） |
| `inference_dongfeng.py` | 东风数据 |
| `inference_new_trajectory.py` | 路侧新轨迹数据（`{scene}_id{track}_seg{NN}` 命名） |
| `inference_extrinsic.py` | 外参扰动实验数据（每实验一个独立文件夹） |
| `inference_control_dropout.py` | **模态剔除探针**：推理时在潜空间把指定模态置零（`--drop_controls hdmap/blur/depth`） |
| `inference_ablation*.py` | 消融模型推理 |

### 启动与工具（`scripts/`）

| 脚本 | 用途 |
|---|---|
| `train_custom_multi_control.sh` | 天津全量训练（输出 /mnt） |
| `train_tianjin_small.sh` | 天津 small（输出 /mnt） |
| `train_dongfeng_multi_control.sh` | 东风训练（输出 /mnt2） |
| `run_dropout_matrix.sh` | 顺序跑 4 个模态剔除推理设定 |
| `eval_dropout_matrix.py` | **独立评测脚本**（零 cosmos 依赖）：PSNR/SSIM/LPIPS/FID/FVD + 配对分析 + 逐段可视化选样排名 |
| `run_eval_matrix_8gpu.sh` | 8 卡并行评测（分片+合并） |
| `count_trainable_params.py` | 从 DCP 检查点元数据统计可训练/冻结参数（不加载权重） |
| `rewrite_dongfeng_captions.py` (experiments/custom 下) | 批量重写东风 caption |
| `merge_segments_for_85frames.py` (同上) | 3×29 帧条件拼 85 帧（4k+1 约束），数据准备工具 |

## 3. 环境

- conda 环境 `cosmos-transfer2`（a100 机器上已配好）；flash-attn 未安装，会有无害警告。
- 国内网络需要的环境变量（各启动脚本已内置）：

```bash
export HF_HOME=/mnt/zihanw/.cache/huggingface
export HF_HUB_CACHE=/mnt/zihanw/.cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
```

- 硬件：8×A100-80G。一个样本（7 视角×29 帧×720p ≈ 20 万 token）由 8 卡 context-parallel 协同处理，**不是数据并行**，故 `WORLD_SIZE=8` 且 batch=1/卡。

## 4. 数据

三批数据，目录规范相同：

```
<数据根>/
├── BlurProjection/
│   ├── captions/ftheta_camera_*/<样本>.json     # 仅 front_wide 的被读取(caption内容+样本索引)
│   ├── control_input_blur/ftheta_camera_*/<样本>.mp4
│   └── videos/ftheta_camera_*/<样本>.mp4        # 真值视频只需在这里有一份
├── DepthSparse/control_input_depth/ftheta_camera_*/<样本>.mp4
└── HDMapBbox/control_input_hdmap_bbox/ftheta_camera_*/<样本>.mp4
```

7 个相机目录：`ftheta_camera_{front_wide_120fov, front_tele_30fov, cross_left_120fov, cross_right_120fov, rear_left_70fov, rear_right_70fov, rear_tele_30fov}`。

| 数据 | 位置 | 规模 | 命名 | 划分 |
|---|---|---|---|---|
| 天津（路口路侧） | `/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data` | 83 场景 × 3 段 | `017_seg01` | 硬编码：75 训 / 8 测（031,033,053,056,076,077,088,089） |
| 东风（自车） | `/mnt2/dongfeng_ego_output/transfer2` | 33 场景、498 段 | `5w_2026-01-12-11-37-14_seg013` | 自动：均匀取 2 场景为测试（`DONGFENG_TEST_SCENES` 可覆盖） |
| 路侧轨迹/外参实验 | 见各推理脚本 docstring | — | `001_id13_seg01` 等 | 仅推理 |

**caption 关键事实**：训练时每视角文本 = 代码内置的视角前缀 + caption 内容；caption 内容**只从 front_wide 相机的 JSON 读取**，其余 6 个相机目录的 JSON 从不被读；caption 里不要写相机朝向。格式 `{"caption": "...", ...}`，其余字段忽略。

## 5. 训练

| 实验名 | 数据 | 起点 | 步数×累积 | 启动 | 输出 |
|---|---|---|---|---|---|
| `custom_multi_control_post_train` | 天津 75 场景 | 自家 iter_5000（warm-restart 态） | 8000×4 | `train_custom_multi_control.sh` | `/mnt/.../cosmos_transfer_custom/multi_control/` |
| `tianjin_multi_control_small` | 天津 10 场景 | 官方检查点 | 400×2 | `train_tianjin_small.sh` | `/mnt/.../cosmos_transfer_custom/tianjin_small/` |
| `dongfeng_multi_control_post_train` | 东风 31 场景 | 官方检查点 | 8000×2 | `train_dongfeng_multi_control.sh` | `/mnt2/Output_R2V_dongfeng/.../dongfeng/` |

启动模板（以东风为例）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 WANDB_MODE=online \
    bash scripts/train_dongfeng_multi_control.sh
```

共同配方：FusedAdamW base lr 8.63e-5（f_max 0.5 → 峰值 ≈4.3e-5，f_min 0.2）、整流流 shift=5 / logit-normal、文本 dropout 0.2、条件帧恒 0、梯度裁剪 0.1。单步约 77-105s（视数据盘 I/O）。

**监控要点**：
- 开训日志核对：场景划分打印、`Freezing base model`、`Found N samples`、检查点加载无大面积 missing keys；
- wandb：训练 loss（每 50 步）、`test_loss/loss`（选检查点看这条）、EMA 采样视频；
- **DeviceMonitor 的 `cpu_mem_gb` 必须平稳**——若线性增长见第 8 节 OOM 事故。

## 6. 推理

模板（天津 small 为例；换数据改 `-m` 模块名）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 \
torchrun --nproc_per_node=8 --master_port=12345 \
  -m cosmos_transfer2.experiments.custom.inference \
  --experiment tianjin_multi_control_small \
  --ckpt_path <run目录>/checkpoints/iter_000000400 \
  --scene_ids 053 077 --context_parallel_size 8 --num_views 7 \
  --output_dir <输出目录>
```

默认采样参数：35 步、guidance 7.0、seed 42、control_weight 1.0、EMA 权重。输出为每样本一个目录，内含 7 个视角的 `*_generated.mp4`。

**模态剔除探针**（验证三路条件的分工）：`inference_control_dropout.py` 加 `--drop_controls hdmap`（可多选），在 VAE 编码后把对应 16 通道潜变量置零，等价于模型的"缺失模态"语义；`scripts/run_dropout_matrix.sh` 一键跑 4 个设定。

## 7. 评估

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_dropout_matrix.py      # 单卡,路径有内置默认
bash scripts/run_eval_matrix_8gpu.sh                               # 8卡并行
```

输出：汇总表（PSNR/SSIM/LPIPS/FID/FVD）、各设定 vs full 的配对差值与胜场数、逐段指标与可视化选样排名，共 3 个 CSV。注意 FID/FVD 用 torchvision Inception/R3D 特征，**仅用于设定间相对比较**，不可与文献绝对值对比。

天津全量模型（iter_6600）的模态剔除结果（7 测试场景 × 7 视角）：

| 设定 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| full | 15.24 | 0.410 | 0.368 | 31.4 | 21.4 |
| no_hdmap | 14.89 | 0.398 | 0.426 | 58.1 | 31.9 |
| no_blur | 13.14 | 0.295 | 0.624 | 64.3 | 36.8 |
| no_depth | 14.99 | 0.406 | 0.389 | 32.6 | 22.4 |
| no_control | 12.47 | 0.281 | 0.693 | 142.3 | 74.7 |

结论：三路条件呈**层级互补**——光度路承载主体信息；布局路的贡献主要在分布级语义保真（剔除后 FID 翻倍而 PSNR 几乎不变，对应其继承的预训练语义能力）；几何路提供小而一致的增益（21 段中 20 段变差）。

## 8. 已知问题与事故记录

- **共享内存 OOM（已修复）**：DataLoader 以 /dev/shm 传递 ~2.5GB/批的视频张量，worker 生灭与常驻 worker-loader 曾导致 1TB 主机内存耗尽、rank 被 OOM killer 杀死、其余 rank NCCL 超时连锁崩溃（表象是 NCCL 报错，根因看 `dmesg | grep -i oom`）。修复：训练 loader 2 常驻 worker；test loss 回调改用临时无 worker loader。若 `cpu_mem_gb` 再出现线性增长，先查这两处；
- **检查点磁盘**：每个 DCP 检查点（含 EMA+优化器）约 20-30GB，长训练注意定期清理早期检查点，只留 test loss 最低的几个；
- **端口分配**：天津全量 12345 / 新轨迹 12347 / 外参 12348 / 东风 12349 / 东风推理 12350 / 天津 small 12351，避免同机冲突；
- **场景号解析**：天津按 `split('_')[0]`，东风按去掉 `_seg\d+$` 后缀——给新数据写配置时注意命名兼容（继承数据集类并重写 `_extract_scene_id` 即可）；
- 首个 iteration 耗时约 2 倍（kernel 编译/缓存预热），属正常。

## 9. 参数统计（论文引用值）

```bash
python3 scripts/count_trainable_params.py <ckpt>/model
```

全模型 2.388B（EMA 副本另计）；可训练 328.8M = 13.77%（控制块 297.8M + hint MLP 29.4M + 控制嵌入器 1.6M）；冻结 2.059B = 86.23%。
