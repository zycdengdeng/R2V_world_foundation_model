# 天津数据 Small 版训练说明

> 实验名 `tianjin_multi_control_small` · 配置文件 `cosmos_transfer2/experiments/custom/tianjin_small_experiment.py` · 启动脚本 `scripts/train_tianjin_small.sh`

## 1. 这个实验是什么

在天津（路口路侧）数据的**缩减子集**上，从 NVIDIA 官方 Cosmos-Transfer2.5 检查点出发做一次**短周期**的多控制后训练。定位是快速验证 / 数据量对比实验，不是正式模型：

- 训练数据从全量的 75 场景缩到 **10 场景**（约 30 个 29 帧片段）；
- 训练步数从全量的 8000 缩到 **400 步**；
- **测试集与全量模型完全一致**（8 个场景），因此本实验的 test loss 与推理指标可直接和全量模型（iter_6600）对比——配方中除数据量与步数外无任何其他差异。

预计耗时：按 105s/iter 估算，400 步 ≈ **12.5 小时**（含评估与采样开销）。

## 2. 数据划分

| 集合 | 场景 | 片段数 |
|---|---|---|
| 训练（10 个，75 个中均匀间隔选取） | 001, 009, 019, 027, 037, 044, 052, 061, 069, 079 | ~30 |
| 测试（与全量运行相同的 8 个） | 031, 033, 053, 056, 076, 077, 088, 089 | ~24 |

- 手动指定训练子集：启动时加环境变量 `TIANJIN_TRAIN_SCENES="001,005,..."`；
- 其余 57 个训练场景在本实验中**不参与任何环节**。

## 3. 基于的权重

**NVIDIA 官方 Cosmos-Transfer2.5-2B Auto Multiview 检查点**（checkpoint-db uuid `4ecc66e9-df19-4aed-9802-0d11e057287a`，HuggingFace `nvidia/Cosmos-Transfer2.5-2B` auto/multiview），即带预训练 hdmap_bbox 控制头的多视图模型。只加载权重、不加载优化器状态（全新训练）。

与东风版、天津全量第一阶段的起点相同，保证三者可比。

## 4. 超参数

### 模型与控制（与全量/东风配方一致）
| 项 | 值 |
|---|---|
| 网络 | cosmos_v1_2B_multiview_control（28 块 DiT，2048 维，16 头） |
| 控制通道 | `hint_keys="hdmap_blur_depth"`，潜通道 0-15 / 16-31 / 32-47 |
| 控制块 | 4 块（spaced，every_n=7），配对主干第 0/7/14/21 层 |
| 可训练参数 | 控制嵌入器 + input hint MLP + 4 控制块 = 329M（13.8%）；主干/VAE/文本编码器/相机嵌入冻结 |
| 条件帧 | 恒为 0（纯控制驱动生成） |
| 视角 | 7 视角联合训练；caption 仅 front_wide + 自动视角前缀，文本 dropout 0.2 |
| 分辨率/帧 | 720×1280，29 帧，10 fps，state_t=8（Wan2.1 tokenizer） |

### 优化与日程（本实验特有的缩短版）
| 项 | 值 |
|---|---|
| 总步数 | **400**（梯度累积 2 → 800 样本 ≈ 27 epoch） |
| 优化器 | FusedAdamW，base lr 8.63e-5，wd 1e-3，betas (0.9, 0.999) |
| lr 调度 | warmup **50** 步 → 峰值 ≈4.3e-5（f_max 0.5）→ 衰减至 ≈1.7e-5（f_min 0.2），cycle **400** |
| 梯度裁剪 | 0.1 |
| 扩散 | 整流流，shift=5，logit-normal 时间采样，uniform 权重 |
| 并行 | 8 GPU，context parallel 8，FSDP shard 8，batch 1/卡 |

### 回调与存档
| 项 | 值 |
|---|---|
| 检查点 | **每 100 步**（共 4 个，DCP 格式含 EMA+优化器） |
| test loss | 每 **200** 步（全 8 测试场景 × 4 时间步采样；临时无 worker loader） |
| EMA 可视化采样 | 每 **200** 步（35 步采样 / guidance 7 / control weight 1.0） |
| 文本嵌入缓存 | 开启（CPU 存储，上限 40 组——天津 caption 按场景不同） |
| DataLoader | num_workers=2，persistent_workers=True（防共享内存泄漏） |

## 5. 怎么跑

```bash
cd /mnt/zihanw/R2V_world_foundation_model_v1
git pull
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 WANDB_MODE=online \
    bash scripts/train_tianjin_small.sh
```

输出目录：`/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/tianjin_small/2b_tianjin_small_<时间戳>/`

开训后核对日志：
1. `[tianjin_small] Train subset: 10/75 scenes: ['001', '009', ...]`；
2. `Freezing base model`；
3. `[MultiControlMultiviewDataset] Found ~30 samples`；
4. DeviceMonitor 的 `cpu_mem_gb` 保持平稳（不再出现 OOM 泄漏那种线性增长）。

## 6. 怎么看结果

- **过程中**：iter 200 / 400 的 EMA 采样视频（run 目录下采样文件夹 + wandb），直观判断小数据短训能到什么程度；test loss 两个点与全量 run 的同期曲线对比；
- **训完后**：用 `cosmos_transfer2.experiments.custom.inference`（旧数据推理脚本本就指向天津数据与同一测试集）加 `--experiment tianjin_multi_control_small --ckpt_path .../iter_000000400` 对 8 个测试场景推理，再用 `scripts/eval_dropout_matrix.py` 的指标口径与全量模型（iter_6600）比较 PSNR/SSIM/LPIPS/FID/FVD。

## 7. 可调旋钮

| 想要 | 改哪里 |
|---|---|
| 压到 ~8 小时 | `max_iter=250`、`cycle_lengths=[250]` |
| 更多/更少数据 | `NUM_SMALL_TRAIN_SCENES`，或 `TIANJIN_TRAIN_SCENES` 环境变量 |
| 换训练起点为自家全量模型 | `checkpoint.load_path` 指向 iter_6600（注意：那是"继续训练"实验，不再是数据量对比） |
