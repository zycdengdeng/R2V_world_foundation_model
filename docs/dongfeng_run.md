# 东风数据训练说明

> 实验名 `dongfeng_multi_control_post_train` · 配置文件 `cosmos_transfer2/experiments/custom/dongfeng_multi_control_experiment.py` · 启动脚本 `scripts/train_dongfeng_multi_control.sh`

## 1. 这个实验是什么

把多控制后训练配方应用到一批全新的自车采集数据（东风）上，从 NVIDIA 官方 Cosmos-Transfer2.5 检查点全新训练，验证配方的跨数据集可迁移性。与天津配方的差异仅在数据、步数与评估间隔，模型与优化配方逐项相同。

## 2. 数据

| 项 | 值 |
|---|---|
| 数据根 | `/mnt2/dongfeng_ego_output/transfer2`（BlurProjection / DepthSparse / HDMapBbox 三目录规范） |
| 规模 | 33 个场景（drive）、498 个片段；7 视角 × 29 帧 × 720p，10 fps |
| 命名 | `{drive}_seg{NNN}`，如 `5w_2026-01-12-11-37-14_seg013`（场景号 = 去掉 `_segNNN` 后缀） |
| 数据集类 | `DongfengMultiControlDataset`（仅重写场景号解析，其余继承天津数据集类） |

**划分（自动）**：模块加载时扫描 front_wide caption 目录得到场景列表，排序后**均匀取 2 个为测试**：

- 测试：`5k_2026-01-12-10-35-46`、`5w_2026-01-12-11-29-05`（~30 段）
- 训练：其余 31 个场景（~468 段）
- 覆盖方式：环境变量 `DONGFENG_TEST_SCENES="场景A,场景B"`（训练与推理需保持一致）

**caption**：全数据集统一的通用场景描述（白天中国城市道路/多车道/路侧建筑树木/混合交通），由 `rewrite_dongfeng_captions.py` 批量写入；只有 front_wide 相机目录的 JSON 被读取，视角前缀由代码自动拼接。

## 3. 基于的权重

**NVIDIA 官方 Cosmos-Transfer2.5-2B Auto Multiview**（checkpoint-db uuid `4ecc66e9-df19-4aed-9802-0d11e057287a`，带预训练 hdmap_bbox 控制头）。`load_training_state=False`，只加载权重、优化器全新。与天津全量第一阶段、天津 small 同起点。

## 4. 超参数

### 模型与控制（全系配方一致项）
| 项 | 值 |
|---|---|
| 网络 | cosmos_v1_2B_multiview_control（28 块 DiT，2048 维，16 头） |
| 控制通道 | `hint_keys="hdmap_blur_depth"` → 潜通道 0-15 / 16-31 / 32-47（布局槽继承预训练权重） |
| 控制块 | 4 块（spaced，every_n=7），配对主干第 0/7/14/21 层，零初始化投影注入 |
| 可训练参数 | 329M（13.8%）：控制嵌入器 + input hint MLP + 4 控制块；主干/VAE/文本编码器/相机嵌入冻结 |
| 条件帧 | 恒 0（纯控制驱动生成） |
| 视角/分辨率 | 7 视角联合（context parallel 8），720×1280 × 29 帧，state_t=8（Wan2.1 tokenizer） |
| 文本 | reason1p1_7B 在线编码，dropout 0.2；文本嵌入缓存开启（caption 全集相同 → 实际仅 1 条缓存，7B 编码器首步后不再运行） |

### 优化与日程
| 项 | 值 |
|---|---|
| 总步数 | **8000**（梯度累积 **2** → 有效 batch 4 样本/步，16000 样本 ≈ 34 epoch） |
| 优化器 | FusedAdamW，base lr 8.63e-5，weight_decay 1e-3，betas (0.9, 0.999) |
| lr 调度 | warmup **500** 步 → 峰值 ≈4.3e-5（f_max 0.5）→ 衰减至 ≈1.7e-5（f_min 0.2），cycle 8000 |
| 梯度裁剪 | 0.1 |
| 扩散 | 整流流，shift=5，logit-normal 时间采样，uniform 时间权重 |
| 并行 | 8×A100-80G，context_parallel_size=8，fsdp_shard_size=8，batch 1/卡 |
| 单步耗时 | 实测 ~77s → 纯训练约 **7 天** |

### 回调与存档
| 项 | 值 |
|---|---|
| 检查点 | 每 **200** 步（共 40 个，DCP 格式含 EMA+优化器，每个 ~20-30GB——**注意磁盘，建议中途清理早期检查点，保留 test loss 最低的几个**） |
| test loss | 每 **400** 步（2 测试场景 ~30 段 × 4 时间步；临时无 worker loader） |
| EMA 可视化采样 | 每 **500** 步（35 步采样 / guidance 7 / control weight 1.0） |
| DataLoader | num_workers=2，persistent_workers=True（共享内存泄漏修复） |
| 分配器 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（存档时显存碎片化缓解） |

## 5. 怎么跑

```bash
cd /mnt/zihanw/R2V_world_foundation_model_v1
git pull
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 WANDB_MODE=online \
    bash scripts/train_dongfeng_multi_control.sh
```

输出：`/mnt2/Output_R2V_dongfeng/cosmos_transfer_custom/dongfeng/2b_dongfeng_multi_control_<时间戳>/`（主端口 12349）。

开训核对日志：
1. `[dongfeng_multi_control] 33 scenes total; 31 train / 2 test. Test scenes: [...]`；
2. `Freezing base model`；
3. `Found N samples`（N ≈ 460+，帧数校验会跳过少量样本属正常）；
4. `[CacheTextEmbeddings] installed` → 首步 `cached new caption set` → 之后全部 `cache hits`；
5. DeviceMonitor 的 `cpu_mem_gb` **全程平稳**（若线性增长见第 7 节）。

## 6. 推理与评估

训练完成后用东风专用推理脚本（默认推理测试集全部样本）：

```bash
torchrun --nproc_per_node=8 --master_port=12350 \
  -m cosmos_transfer2.experiments.custom.inference_dongfeng \
  --ckpt_path <run目录>/checkpoints/iter_00XXXX \
  --output_dir /mnt2/Output_R2V_dongfeng/inference \
  --context_parallel_size 8 --num_views 7
```

采样参数：35 步、guidance 7、seed 42、EMA 权重。检查点选择看 wandb 的 `test_loss/loss` 曲线最低点。指标评测可复用 `scripts/eval_dropout_matrix.py` 的口径（`--gt_videos_root` 指向东风的 `BlurProjection/videos`，`--results` 指向推理输出）。

## 7. 事故记录与加固（本 run 淌出来的坑）

- **主机共享内存泄漏 → OOM → NCCL 连锁崩溃**（首次运行 iter ~330）：DataLoader 经 /dev/shm 传 ~2.5GB/批的视频张量，worker 生灭与评估回调的常驻 worker-loader 累积泄漏，耗尽 1TB 主机内存后 rank 被 OOM killer 杀死，表象是 NCCL 超时。诊断：`dmesg | grep -i oom`。修复：训练 loader 2 常驻 worker、test loss 回调改临时无 worker loader（已沉淀进代码）；
- **存档时显存碎片化 OOM**（天津 small 首存崩溃,同配方风险）：DCP 保存的聚合分配在碎片化显存上偶发失败,表象 `NCCL Error 1: unhandled cuda error`。修复：`expandable_segments` 分配器开关（已进启动脚本）；
- 首个 iteration 耗时 ~2 倍属正常（kernel 编译预热）。

## 8. 可调旋钮

| 想要 | 改哪里 |
|---|---|
| 换测试场景 | `DONGFENG_TEST_SCENES` 环境变量（训练与推理保持一致） |
| 提前停 | test loss 走平即可手动停,每 200 步都有检查点 |
| 换输出盘 | 启动时覆盖 `IMAGINAIRE_OUTPUT_ROOT` |
