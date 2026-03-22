Teacher Model 端到端验真结论
验真结果：Teacher 机制已真实接通
通过 A/B 对照实验，从日志中确认了完整链路的每一个环节。

A/B 对照日志证据
Teacher-OFF（对照组）
检查点	结果
teacher model 创建	teacher_pretrain not set -- teacher model DISABLED
teacher_generator	None
experience_maker.teacher_samples_generator	None
每步 teacher branch gate	cf_target_mode=single, teacher_samples_generator=None => SKIP teacher branch
teacher_embedding 构建	无
_build_cf_target_embedding	未触发
reward 值 (4步)	0.0211, 0.0206, 0.0217, 0.0204
Teacher-ON（实验组）
检查点	结果
teacher model 创建	Creating teacher RayActorGroup from: /root/code/Qwen3.5-2B
参数传递	cf_target_mode=teacher, distribution_reward_type=cf_l1oo, cf_teacher_lambda=0.5, cf_teacher_n_samples=2
teacher_generator	PRESENT
experience_maker.teacher_samples_generator	PRESENT
teacher 权重加载	Initializing teacher weights from: /root/code/Qwen3.5-2B
每步 teacher branch gate	cf_target_mode=teacher, teacher_samples_generator=PRESENT => ENTER teacher branch
teacher_embedding 构建	shape=torch.Size([1, 1, 2, 15, 6144]), mean=-0.000291
_build_cf_target_embedding	MIXED target built! lambda=0.5, r_gt=2, m_teacher=2, target_shape=(1,1,4,15,6144) (vs GT-only would be (1,1,1,15,6144))
reward 值 (4步)	0.0333, 0.0336, 0.0305, 0.0313
关键差异（证明 teacher 真实参与）
分支选择不同：OFF → SKIP，ON → ENTER
teacher_embedding 有实际数值：shape [1,1,2,15,6144]，mean ≈ -0.0003（非零非 NaN）
target distribution shape 不同：ON 的 target (1,1,4,15,6144) vs GT-only (1,1,1,15,6144)，4 = 2 GT copies + 2 teacher samples（λ=0.5 时 r=2）
reward 数值系统性不同：ON ≈ 0.032，OFF ≈ 0.021，差异 ~50%（不是随机波动）
每步都一致触发：4 个 global step 每步都有 ENTER + teacher_embedding built + MIXED target built
修复的 bug（验真过程中发现并修复）
Bug	位置	修复
NCCL 端口冲突	launcher.py::_get_free_port	移除 preferred port 逻辑，改为纯 OS 随机分配
Teacher 并发初始化	train_ebft_ray.py::train()	Teacher 在 actor 之后串行初始化
Teacher 缺少 max_steps	train_ebft_ray.py::train()	传入 max_steps 参数
datatrove 硬依赖	ebft_trainer.py	改为 try/except 可选导入
使用脚本
脚本路径：scripts/run_teacher_verify.sh

# Teacher-OFF 对照运行
TEACHER_ENABLE=0 MODEL_PATH=/root/code/Qwen3.5-2B \
  bash scripts/run_teacher_verify.sh
# Teacher-ON 验真运行
TEACHER_ENABLE=1 MODEL_PATH=/root/code/Qwen3.5-2B \
  bash scripts/run_teacher_verify.sh
可配置环境变量：MODEL_PATH、TEACHER_ENABLE、CUDA_VISIBLE_DEVICES、RUN_ROOT。