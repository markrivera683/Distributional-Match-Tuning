# G2 脚本分类目录

本文档对 `scripts/` 目录下与 G2 阶段相关的脚本进行分类，帮助你快速找到需要运行或参考的文件。

---

## 1. 正式训练脚本（8 卡 A100）

这些是用于在 8x A100 上执行完整 G2 训练的生产脚本。

### 1.1 推荐：`run_g2_8gpu_remote_teacher.sh`


| 属性  | 值                                                                 |
| --- | ----------------------------------------------------------------- |
| 路径  | `scripts/run_g2_8gpu_remote_teacher.sh`                           |
| GPU | 8x A100，actor+ref 4卡 / critic+reward 4卡                           |
| 奖励  | `cf_l1oo` + `cf_target_mode=teacher`                              |
| 教师  | remote（HTTP API）                                                  |
| 特点  | 所有变量集中在顶部 8 个编号区块；支持环境变量覆盖；可选 teacher cache；有 preflight 检查和启动信息面板 |
| 输出  | `RUN_ROOT` 含时间戳，每次自动创建新目录                                         |


**推荐作为 G2 正式训练的首选脚本。**

### 1.2 Baseline 复跑：`run_g2_baseline_8gpu_rerun.sh`


| 属性  | 值                                                                 |
| --- | ----------------------------------------------------------------- |
| 路径  | `scripts/run_g2_baseline_8gpu_rerun.sh`                           |
| GPU | 8x A100，actor+ref 4卡 / critic+reward 4卡                           |
| 奖励  | `cf_l1oo` + `cf_target_mode=teacher`                              |
| 教师  | remote（HTTP API）                                                  |
| 特点  | 与论文/实验对齐的一套固定超参；始终启用 `--teacher_cache_enable`；变量部分硬编码（不可通过环境变量覆盖） |
| 用途  | 精确复现特定实验结果                                                        |


### 1.3 自动 Batch 约束版：`run_train_qwen35_2b_aops_g2_remote_teacher.sh`


| 属性  | 值                                                       |
| --- | ------------------------------------------------------- |
| 路径  | `scripts/run_train_qwen35_2b_aops_g2_remote_teacher.sh` |
| GPU | 8x A100（默认 actor/critic 各 3 卡 + ref/reward 各 1 卡）       |
| 奖励  | `cf_l1oo` + `cf_target_mode=teacher`                    |
| 教师  | remote（HTTP API）                                        |
| 特点  | 脚本较长（~400 行）；包含自动 DeepSpeed/ED 双约束 batch size 调整逻辑      |
| 用途  | 当 batch 配置需要自动校验和调整时使用                                  |


---

## 2. 冒烟测试脚本

用于快速（< 5 分钟）验证远程教师→目标测度→奖励计算全链路是否正常工作。

### 2.1 `run_g2_remote_teacher_smoke.sh`


| 属性    | 值                                                                                              |
| ----- | ---------------------------------------------------------------------------------------------- |
| 路径    | `scripts/run_g2_remote_teacher_smoke.sh`                                                       |
| GPU   | 单卡（默认 `CUDA_VISIBLE_DEVICES=0`），`--colocate_all_models`                                        |
| 预算    | `max_samples=4`，`num_episodes=1`，`rollout_batch_size=1`                                        |
| 用途    | 验证 remote teacher API 连通、teacher 回答写入 target、reward 非零                                         |
| 日志检查点 | 脚本头注释列出了关键日志标记：`[RemoteTeacher] Init`、`[TEACHER-TARGET] MIXED target built`、`rewards mean=...` |


---

## 3. 单机 / 少卡开发脚本

适合在单张 GPU 上进行代码调试和小规模功能验证。

### 3.1 `run_train_local_qwen35_2b_aops_remote_teacher.sh`


| 属性  | 值                                                          |
| --- | ---------------------------------------------------------- |
| 路径  | `scripts/run_train_local_qwen35_2b_aops_remote_teacher.sh` |
| GPU | 单卡（默认 `CUDA_VISIBLE_DEVICES=0`），各组件 1 GPU                  |
| 奖励  | `cf_l1oo` + `cf_target_mode=teacher`                       |
| 教师  | remote（HTTP API）                                           |
| 用途  | 单机日常开发迭代，不用占满 8 卡                                          |


---

## 4. 辅助工具脚本（非训练主流程）


| 脚本                                  | 类型        | 用途                                           |
| ----------------------------------- | --------- | -------------------------------------------- |
| `scripts/test_teacher_provider.py`  | Python 单测 | 测试 `teacher_provider.py` 的 cache 和 HTTP 调用逻辑 |
| `scripts/mock_teacher_server.py`    | Python 服务 | 本地假 OpenAI 兼容教师服务，用于无真实教师时的端到端测试             |
| `scripts/case_study_aops.py`        | Python 分析 | AoPS 基座模型 case study 和评测审计                   |
| `scripts/summarize_aops_results.py` | Python 分析 | 汇总 AoPS 结果                                   |
| `scripts/evaluate_reward_ce.py`     | Python 评测 | 奖励与交叉熵评估                                     |
| `scripts/ebft_sweep.py`             | Python 超参 | 超参扫描辅助                                       |


---

## 5. 历史 / 异构环境脚本（不推荐直接使用）

这些脚本来自早期开发阶段或其他机器配置，路径和模型与当前环境不匹配，**不推荐直接照抄运行**。仅供了解历史实验配置。


| 脚本                                                          | 环境差异                                                       | 说明                                  |
| ----------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------- |
| `scripts/run_aops_g2_8gpu.sh`                               | 路径 `/mnt/workspace/...`，模型 `Qwen2.5-1.5B`，无 remote teacher | 旧 bundle 的 cf_l1oo 纯分布跑法            |
| `scripts/run_aops_g1_8gpu.sh`                               | 路径 `/mnt/workspace/...`                                    | 旧 G1 配置                             |
| `scripts/run_aops_g3_8gpu.sh`                               | 路径 `/mnt/workspace/...`                                    | 旧 G3 配置（含 feature adapter，超出 G2 范围） |
| `scripts/run_aops_group1_baseline.sh`                       | 路径 `/mnt/workspace/...`                                    | 旧 group1 baseline                   |
| `scripts/run_aops_group2_distonly.sh`                       | 路径 `/mnt/workspace/...`                                    | 旧 group2 纯分布                        |
| `scripts/run_aops_group3_distema.sh`                        | 路径 `/mnt/workspace/...`                                    | 旧 group3 dist+EMA                   |
| `scripts/run_aops_baseline.sh`                              | 路径 `/mnt/workspace/...`                                    | 旧 baseline                          |
| `scripts/run_aops_cf_l1oo.sh`                               | 路径 `/mnt/workspace/...`                                    | 旧 cf_l1oo 独立测试                      |
| `scripts/run_aops_triplet.sh` / `run_aops_triplet_night.sh` | 路径 `/mnt/workspace/...`                                    | triplet 实验                          |
| `scripts/g1_formal_8gpu_real.sh` / `v2` / `v3`              | 路径 `/mnt/workspace/...`                                    | G1 正式多版本迭代                          |
| `scripts/dry_run_8gpu.sh` / `v2` / `v3` / `v4`              | 路径 `/mnt/workspace/...`                                    | 早期 dry run                          |
| `scripts/run_step0_vanilla_qwen3_smoke.sh`                  | 路径不定                                                       | Step0 vanilla 冒烟                    |
| `scripts/run_step1_*.sh`                                    | 路径不定                                                       | Step1 feature map 实验                |
| `scripts/run_step2lite_*.sh`                                | 路径不定                                                       | Step2 adapter 实验                    |
| `scripts/run_u1a_rff_qwen3_smoke.sh`                        | 路径不定                                                       | U1a RFF 冒烟                          |
| `scripts/run_ebft_local_*.sh`                               | 路径不定                                                       | 早期本地 EBFT 测试                        |
| `scripts/run_ebft_example.sh`                               | 路径不定                                                       | 示例脚本                                |
| `scripts/setup_env.sh`                                      | 旧环境 setup                                                  | 旧环境初始化                              |
| `scripts/create_ebft_transfer_bundle.sh`                    | 打包工具                                                       | 迁移用打包脚本                             |
| `scripts/start_triplet_night_tmux.sh`                       | tmux 辅助                                                    | 夜间 triplet 批量运行                     |
| `scripts/run_teacher_verify.sh`                             | 教师验证                                                       | 早期教师端点验证                            |
| `scripts/g3_precheck.sh`                                    | G3 预检                                                      | G3 阶段预检，超出 G2 范围                    |


---

## 6. G1 相关脚本（非 G2，仅供对比）


| 脚本                                       | 说明                                                                          |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| `scripts/run_g1_baseline_8gpu_rerun.sh`  | G1 baseline 复跑，`distribution_reward_type=pointwise`，使用 GT 答案作为 target，无远程教师 |
| `scripts/eval_g1_baseline_8gpu_rerun.sh` | G1 baseline 评测脚本                                                            |


---

## 7. 框架示例脚本（`examples/scripts/`）

`examples/scripts/` 目录下是 OpenRLHF 框架自带的训练示例，涵盖 PPO、DPO、GRPO、SFT、Rejection Sampling 等标准流程。这些脚本与 G2 无直接关系，但对理解框架能力有帮助。

---

## 速查：「我应该用哪个脚本？」


| 场景               | 推荐脚本                                                   |
| ---------------- | ------------------------------------------------------ |
| 首次验证环境是否能跑通      | `run_g2_remote_teacher_smoke.sh`                       |
| 单卡开发调试           | `run_train_local_qwen35_2b_aops_remote_teacher.sh`     |
| 8 卡正式 G2 训练      | `run_g2_8gpu_remote_teacher.sh`                        |
| 精确复现已有实验         | `run_g2_baseline_8gpu_rerun.sh`                        |
| 不确定 batch 配置是否合法 | `run_train_qwen35_2b_aops_g2_remote_teacher.sh`（含自动校验） |
| 无真实教师 API，纯本地测试  | 先启动 `mock_teacher_server.py`，再跑 smoke                  |
| 与 G1 baseline 对比 | 同时参考 `run_g1_baseline_8gpu_rerun.sh`                   |


