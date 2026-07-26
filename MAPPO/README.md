# MAPPO on MPE Simple Spread

这是一个适合课程展示与 GitHub 托管的精简版 MAPPO 项目。代码从原始
`on-policy` 工程中人工筛选，保留了 MPE Simple Spread 上的训练、评估和
行为分析闭环，并维持原 `onpolicy` 包结构以保证 import 路径一致。

## 项目背景

- 原论文：**The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**
- 实验环境：Multi-Agent Particle Environment (MPE) `simple_spread`
- Agent 数量：3
- Landmark 数量：3
- 基线：共享策略 MAPPO baseline
- 改进：在 Simple Spread 奖励中加入可配置的 duplicate penalty，抑制多个
  Agent 重复占据同一 Landmark
- 行为评测：episode reward、coverage、duplicate occupancy、collision 和
  trajectory

duplicate penalty 的配置入口位于 `onpolicy/config.py`，环境奖励与
duplicate occupancy 信息位于
`onpolicy/envs/mpe/scenarios/simple_spread.py`。训练期间的覆盖、碰撞和重复
占据统计由 `onpolicy/runner/shared/mpe_runner.py` 汇总。

## 环境安装

建议使用 Python 3.8（原项目依赖较旧版 Gym）。

```powershell
cd C:\Users\20392\Desktop\MPE\MAPPO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

在 Linux/macOS 中，将激活命令替换为 `source .venv/bin/activate`。

## 训练

MAPPO baseline：

```powershell
python -m onpolicy.scripts.train.train_mpe --env_name MPE --scenario_name simple_spread --algorithm_name mappo --experiment_name baseline --num_agents 3 --num_landmarks 3 --seed 1 --n_training_threads 1 --n_rollout_threads 8 --episode_length 25 --num_env_steps 1000000 --ppo_epoch 10 --num_mini_batch 1 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name student
```

启用 duplicate penalty：

```powershell
python -m onpolicy.scripts.train.train_mpe --env_name MPE --scenario_name simple_spread --algorithm_name mappo --experiment_name duplicate_penalty --num_agents 3 --num_landmarks 3 --seed 1 --n_training_threads 1 --n_rollout_threads 8 --episode_length 25 --num_env_steps 1000000 --ppo_epoch 10 --num_mini_batch 1 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_duplicate_penalty --duplicate_penalty 0.1 --duplicate_threshold 0.1 --user_name student
```

训练脚本默认将实验输出放到 `onpolicy/scripts/results/`；该目录已被
`.gitignore` 排除。

## 评估

行为指标评估（模型目录应包含 `actor.pt` 与 `critic.pt`）：

```powershell
python -m onpolicy.scripts.evaluate_agent_behavior --model_name baseline --model_dir path\to\model --output_dir evaluation\baseline --num_eval_episodes 100
```

随机策略 sanity check：

```powershell
python -m onpolicy.scripts.evaluate_random_policy --num_eval_episodes 100 --output_dir evaluation\random
```

对比两组或多组行为评估 CSV：

```powershell
python -m onpolicy.scripts.compare_agent_behavior --csv_files evaluation\baseline\episode_metrics.csv evaluation\penalty\episode_metrics.csv --output_dir evaluation\comparison
```

绘制单回合轨迹：

```powershell
python -m onpolicy.scripts.plot_agent_trajectory --model_name baseline --model_dir path\to\model
```

各脚本的完整参数可使用 `python -m <模块名> --help` 查看。

## 目录说明

```text
MAPPO/
├── docs/                         课程报告与运行手册
├── onpolicy/
│   ├── algorithms/r_mappo/       MAPPO Trainer、Policy、Actor/Critic
│   ├── algorithms/utils/         网络层、动作分布与 RNN 工具
│   ├── envs/mpe/                 MPE 与改进后的 Simple Spread
│   ├── runner/shared/            共享策略训练与指标汇总流程
│   ├── scripts/                  训练、评估、对比与轨迹脚本
│   ├── utils/shared_buffer.py    rollout/replay buffer
│   └── config.py                 训练及 duplicate penalty 配置
├── results/                      少量代表性图片与分析报告
├── requirements.txt
└── .gitignore
```

本精简版不包含训练权重、日志、虚拟环境、缓存、论文 PDF 或与 MPE MAPPO
无关的第三方环境源码。评估训练模型时，请自行将权重放在本地路径中；权重
扩展名已默认被 Git 忽略。

> 范围说明：本仓库仅支持 README 中展示的 shared-policy MAPPO 路径。
> 原入口中为兼容上游工程而保留的 MAT 与 separated-policy 条件分支未随精简版
> 一同打包。
