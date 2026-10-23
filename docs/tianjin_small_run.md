# 天津数据 Small 版训练说明（可直接照跑）

> 实验名 `tianjin_multi_control_small` · 配置 `cosmos_transfer2/experiments/custom/tianjin_small_experiment.py` · 启动脚本 `scripts/train_tianjin_small.sh`
> 机器：a100（8×A100-80G）· 代码目录 `/mnt/zihanw/R2V_world_foundation_model_v1` · 分支 `claude/review-and-continue-bpISK`

## 0. 上机准备

```bash
# 登录 a100 后，进入（或新建）tmux 会话——长任务一律在 tmux 里跑
tmux attach -t small        # 会话已存在时
# tmux new -s small         # 不存在时新建；挂后台: Ctrl+B 然后按 D

cd /mnt/zihanw/R2V_world_foundation_model_v1
git pull                    # 确认在分支 claude/review-and-continue-bpISK 上
```

## 1. 这个实验是什么

在天津（路口路侧）数据的**缩减子集**上，从 NVIDIA 官方 Cosmos-Transfer2.5 检查点出发做**短周期**多控制后训练。定位是快速验证 / 数据量对比实验：

- 训练数据：全量 75 场景 → **10 场景**（约 30 个 29 帧片段）；
- 步数：全量 8000 → **400 步**；
- 测试集：**053、077** 两个场景（全量模型也没训过它们,指标与全量模型 iter_6600 直接可比）。

实测约 77s/iter,总耗时 **9-10 小时**。

## 2. 数据位置与划分

数据根（所有路径都在这底下）：

```
/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/
├── BlurProjection/
│   ├── captions/ftheta_camera_front_wide_120fov/   ← caption（只有这个相机的被读取）
│   ├── control_input_blur/ftheta_camera_<相机>/     ← 光度条件
│   └── videos/ftheta_camera_<相机>/                 ← 真值视频
├── DepthSparse/control_input_depth/ftheta_camera_<相机>/    ← 几何条件
└── HDMapBbox/control_input_hdmap_bbox/ftheta_camera_<相机>/ ← 布局条件
```

7 个相机目录名：`ftheta_camera_{front_wide_120fov, front_tele_30fov, cross_left_120fov, cross_right_120fov, rear_left_70fov, rear_right_70fov, rear_tele_30fov}`。

| 集合 | 场景 | 片段 |
|---|---|---|
| 训练（10 个,75 个全量训练场景中均匀选取,代码自动完成） | 001, 009, 019, 027, 037, 044, 052, 061, 069, 079 | 每场景 seg01-03,共 ~30 段 |
| 测试 | **053, 077** | ~6 段 |

文件示例（一个测试样本的全部输入）：

```
.../BlurProjection/videos/ftheta_camera_front_wide_120fov/053_seg01.mp4      # 真值
.../BlurProjection/control_input_blur/ftheta_camera_front_wide_120fov/053_seg01.mp4
.../DepthSparse/control_input_depth/ftheta_camera_front_wide_120fov/053_seg01.mp4
.../HDMapBbox/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/053_seg01.mp4
.../BlurProjection/captions/ftheta_camera_front_wide_120fov/053_seg01.json   # 文本
```

其余场景（全量的另外 65 训练场景 + 6 测试场景）在本实验中完全不使用。划分逻辑写在配置文件里,不需要手动准备任何文件。

## 3. 基于的权重

**NVIDIA 官方 Cosmos-Transfer2.5-2B Auto Multiview**（带预训练 hdmap_bbox 控制头,checkpoint-db uuid `4ecc66e9-df19-4aed-9802-0d11e057287a`）,首次运行自动从 HF 镜像下载到 `/mnt/zihanw/.cache/huggingface`。只加载权重、优化器全新。主干冻结,只训控制分支（329M,13.8%）。

## 4. 训练（如需重跑）

> **注意**：已有一次完成的 run：
> `/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/2b_tianjin_small_20261022_110117/`
> 检查点在其 `checkpoints/iter_000000{100,200,300,400}/` 下。**只想看结果的话跳过本节,直接看第 6 节推理。**

```bash
# 在 tmux 里（见第0节）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 WANDB_MODE=online \
    bash scripts/train_tianjin_small.sh
```

新 run 输出到 `/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/2b_tianjin_small_<新时间戳>/`。

开训后前几分钟核对日志：
1. `[tianjin_small] Train subset: 10/75 scenes: ['001', '009', ...]`
2. `[tianjin_small] Small test set (loss + inference): ['053', '077']`
3. `Freezing base model`
4. `Found ~30 samples`
5. DeviceMonitor 的 `cpu_mem_gb` 全程平稳（若线性增长,见 README 事故记录）

### 训练完成后:定位你的 run 和检查点

每次启动训练都会按启动时刻生成一个**新的时间戳目录**,互不覆盖:

```bash
# 列出所有 run,最新的时间戳就是你这次的
ls /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/

# 查看某个 run 存了哪些检查点,以及最新的一个
ls <run目录>/checkpoints/
cat <run目录>/checkpoints/latest_checkpoint.txt
```

**选哪个检查点**:打开该 run 的 wandb 页面（启动日志里有链接）,看 `test_loss/loss` 曲线的 4 个点（iter 100/200/300/400）,取最低点对应的 `iter_000000XXX`。曲线一路下降就取 400;中途见底回升（小数据过拟合的典型形态）就取最低的那个。

**中途崩溃想续训**（而不是重头跑）:改配置文件 `tianjin_small_experiment.py` 的 checkpoint 段——
`load_path` 指向崩溃前最后一个检查点目录（如 `.../checkpoints/iter_000000200`）、`load_training_state=True`,再正常启动;训完记得改回去（`load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path`、`load_training_state=False`）,否则下一次"全新训练"会错误地续上旧状态。

## 5. 超参数

### 模型与控制（与全量/东风配方一致）
| 项 | 值 |
|---|---|
| 网络 | cosmos_v1_2B_multiview_control（28 块 DiT,2048 维） |
| 控制通道 | `hint_keys="hdmap_blur_depth"`,潜通道 0-15 / 16-31 / 32-47 |
| 控制块 | 4 块（spaced every_n=7）,配对主干第 0/7/14/21 层 |
| 可训练 | 329M（13.8%）:控制嵌入器 + hint MLP + 4 控制块;其余冻结 |
| 条件帧 | 恒 0（纯控制驱动）;7 视角联合（CP8）;720×1280×29 帧 |
| 文本 | reason1p1_7B 在线编码 + 0.2 dropout;嵌入缓存开启（上限 40 组） |

### 优化与日程
| 项 | 值 |
|---|---|
| 总步数 | 400（梯度累积 2 → 800 样本 ≈ 27 epoch） |
| 优化器 | FusedAdamW,base lr 8.63e-5,wd 1e-3 |
| lr 调度 | warmup 50 → 峰值 ≈4.3e-5 → 衰减至 ≈1.7e-5,cycle 400 |
| 扩散 | 整流流,shift=5,logit-normal 采样;梯度裁剪 0.1 |
| 检查点 | 每 100 步（4 个）;test loss 每 100 步（与检查点对齐,053+077）;EMA 采样每 200 步 |
| 环境加固 | DataLoader 2 常驻 worker;`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## 6. 推理（完整指令,可直接粘贴）

下面的命令中只有两处需要按你的 run 替换（其余原样照抄）:

| 占位 | 替换为 | 示例中的值 |
|---|---|---|
| `--ckpt_path` | 你的 run 目录 + 选中的检查点（见第 4 节"训练完成后"） | `2b_tianjin_small_20261022_110117/checkpoints/iter_000000300` |
| `--output_dir` | 自定的输出目录,建议带上 iter 号避免不同检查点互相覆盖 | `inference_iter300` |

示例（原作者的 run,iter 300）:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 \
IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1" \
HF_HOME="/mnt/zihanw/.cache/huggingface" \
HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub" \
HF_ENDPOINT="https://hf-mirror.com" \
HF_HUB_ENABLE_HF_TRANSFER=0 \
TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers" \
torchrun --nproc_per_node=8 --master_port=12345 \
  -m cosmos_transfer2.experiments.custom.inference \
  --experiment tianjin_multi_control_small \
  --ckpt_path /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/2b_tianjin_small_20261022_110117/checkpoints/iter_000000300 \
  --output_dir /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/inference_iter300 \
  --context_parallel_size 8 \
  --num_views 7 \
  --scene_ids 053 077
```

采样参数（默认值）:35 步、guidance 7.0、seed 42、control weight 1.0、EMA 权重。6 个样本约 20-30 分钟。**确保此时没有训练占着 8 张卡**（`nvidia-smi` 查看）。

**输出结构**（每样本一个目录,7 个视角各一个生成视频,29 帧/720p/10fps）:

```
.../tianjin_small/inference_iter300/
├── 053_seg01/
│   ├── front_wide_120fov_generated.mp4
│   ├── front_tele_30fov_generated.mp4
│   ├── cross_left_120fov_generated.mp4
│   ├── cross_right_120fov_generated.mp4
│   ├── rear_left_70fov_generated.mp4
│   ├── rear_right_70fov_generated.mp4
│   └── rear_tele_30fov_generated.mp4
├── 053_seg02/  053_seg03/  077_seg01/  077_seg02/  077_seg03/   （同样7个文件）
```

**对照用的真值和基线**:
- 真值视频:`/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection/videos/ftheta_camera_<相机>/053_seg01.mp4`
- 全量模型（iter_6600,同 seed 同采样参数）在 053/077 的生成结果:`/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference/053_seg01/` 等,可直接并排对比,**无需重跑**。

**定量指标**（可选）:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_dropout_matrix.py \
  --gt_videos_root /mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection/videos \
  --results small=/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/inference_iter300 \
            full=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference \
  --output_csv /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/metrics.csv
```

输出 PSNR/SSIM/LPIPS/FID/FVD 对比表（自动取两边共有的样本,即 053/077 的 6 段）。

## 7. 可调旋钮

| 想要 | 改哪里 |
|---|---|
| 更多/更少训练数据 | 配置里 `NUM_SMALL_TRAIN_SCENES`,或启动时 `TIANJIN_TRAIN_SCENES="001,005,..."` |
| 更长/更短训练 | `max_iter` 与 `cycle_lengths` 同步改 |
| 换训练起点为全量模型 | `checkpoint.load_path` 指向 iter_6600（注意:那是"继续训练"实验,不再是数据量对比） |
