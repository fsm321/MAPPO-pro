# MAPPO-pro（greedy 分支）运行说明


当前 `greedy` 分支中，

- 蓝方由固定规则脚本改为**协同贪婪对手**：目标分配 + 候选动作枚举 + 一步状态预测 + 即时效用评分 + `argmax J(a)` 选择动作。
- 训练入口使用 `train_parallel.py`。
- 批量训练入口使用 `run_experiments.py`。
- 训练默认配置已经适配最终实验：`max_episode_steps=256`、`max_train_steps=2.56e8`、`evaluate_freq=1000`、`save_freq=2000`。
- TensorBoard 指标已按 `Performance/`、`Debug/`、`Task/` 分组。

---

## 1. 拉取与切换分支

在代码文件夹右击打开 Git Bash，执行：

```bash
git clone https://github.com/fsm321/MAPPO-pro.git
cd MAPPO-pro
git checkout greedy
git pull origin greedy
```

如果你已经克隆过仓库，只需要执行：

```bash
cd MAPPO-pro
git fetch origin
git checkout greedy
git pull origin greedy
```

---

## 2. 提交代码更新

确认当前分支是 `greedy`：

```bash
git branch
```

提交代码：

```bash
git add .
git status
git commit -m "更新 greedy 分支实验配置"
git push origin greedy
```

如果需要重新跟踪所有文件，可以使用：

```bash
git rm -r --cached .
git add .
git status
git commit -m "更新 greedy 分支实验配置"
git push origin greedy
```

---

## 3. 单独训练 MAPPO / Meta-MAPPO

进入项目根目录，例如：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
```
### 3.1 正式训练 MAPPO

```bash
python train_parallel.py ^
  --algo_name MAPPO ^
  --date MAPPO_greedy_seed10_final
```

### 3.2 正式训练 Meta-MAPPO

```bash
python train_parallel.py ^
  --algo_name Meta-MAPPO ^
  --date Meta_MAPPO_greedy_seed10_final
```

### 3.3 增强版 Meta-MAPPO 训练

如果希望强化 Meta-MAPPO 的 support inner update，可以额外跑一组：

```bash
python train_parallel.py ^
  --algo_name Meta-MAPPO ^
  --seed 10 ^
  --num_envs 8 ^
  --meta_support_envs 4 ^
  --meta_inner_epochs 2 ^
  --meta_outer_epochs 1 ^
  --max_episode_steps 256 ^
  --max_train_steps 256000000 ^
  --evaluate_freq 1000 ^
  --save_freq 2000 ^
  --date Meta_MAPPO_greedy_inner2_seed10_final
```

---

## 4. 批量训练

`run_experiments.py` 会批量运行 MAPPO 和 Meta-MAPPO。当前脚本中：

- `algorithms = ["MAPPO", "Meta-MAPPO"]`
- `all_seeds = [10, 20, 30]`
- `num_parallel_tasks = 2` 时，只跑 seed 10 的两个算法。
- `max_train_steps = int(2.56e8)`。

运行：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
python run_experiments.py
```

---

## 5. TensorBoard 查看训练结果

启动 TensorBoard：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
tensorboard --logdir=./data
```

如果需要从 TensorBoard 导出论文图，在 `result/plot_TensorBoard.py` 中修改数据路径后运行：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\MAPPO-pro\result
python plot_TensorBoard.py
```

---

## 6. 模型评估

评估时需要让 `max_episode_steps` 和训练保持一致，因此统一显式传入：

```bash
--max_episode_steps 256
```

### 6.1 评估 MAPPO

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
python evaluate.py ^
  --algo_name MAPPO ^
  --model_dir ./data/MAPPO_seed10_0507_024934/model/310000
```

### 6.2 评估 Meta-MAPPO

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
python evaluate.py ^
  --algo_name Meta-MAPPO ^
  --model_dir ./data/MetaMAPPO_stable_seed10_lr0005/model/290000
```

评估会生成：

```text
combat_metrics_MAPPO.json
combat_metrics_Meta-MAPPO.json
robustness_data_MAPPO.npy
robustness_data_Meta-MAPPO.npy
recovery_data_MAPPO.npy
recovery_data_Meta-MAPPO.npy
robustness_winrate_MAPPO.npy
robustness_winrate_Meta-MAPPO.npy
robustness_reward_MAPPO.npy
robustness_reward_Meta-MAPPO.npy
```

---

## 7. Meta-test 快速适应评估

快速适应评估用于证明 Meta-MAPPO 的少样本适应能力。运行前先打开：

```text
run_fast_adaptation.py
```

修改模型路径：

```python
MODEL_DIRS = {
    "MAPPO": r"./data/你的MAPPO目录/model/检查点",
    "Meta-MAPPO": r"./data/你的Meta-MAPPO目录/model/检查点",
}
```


然后运行：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
python run_fast_adaptation.py
```

该脚本会调用 `evaluate.py --eval_adaptation --only_adaptation`，对 U0~U4 进行快速适应评估。

结果文件示例：

```text
fast_adaptation_metrics_MAPPO.json
fast_adaptation_metrics_Meta-MAPPO.json
```

---

## 8. 绘制对比图

进入结果目录：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0\result
python plot_combined.py
```

输出文件示例：

```text
./result/combined_combat_metrics.png
./result/combined_robustness.png
./result/combined_recovery.png
```

---

## 9. 3D 轨迹图

示例：

```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\10.0
python plot_3D.py ^
  --algo_name MAPPO ^
  --model_dir ./data/你的MAPPO目录/model/检查点 ^
  --task_id 2 ^
```

---

## 10. 常用自定义参数

MAPPO 常用参数：
```bash
python train_parallel.py ^
  --algo_name MAPPO ^
  --num_envs 8 ^
  --max_train_steps 256000000 ^
  --max_episode_steps 256 ^
  --evaluate_freq 1000 ^
  --save_freq 2000 ^
  --buffer_size 6400 ^
  --batch_size 6400 ^
  --mini_batch_size 1600 ^
  --hidden_width 256 ^
  --lr_a 3e-5 ^
  --lr_c 3e-4 ^
  --epsilon 0.15 ^
  --K_epochs 4
```

Meta-MAPPO 额外常用参数：

```bash
python train_parallel.py ^
  --algo_name Meta-MAPPO ^
  --num_envs 8 ^
  --meta_support_envs 4 ^
  --meta_buffer_size 3200 ^
  --meta_inner_epochs 1 ^
  --meta_outer_epochs 1 
```

主要参数说明：

- `--scenario_name`: 场景名称，当前默认 `air_combat_2v2`。
- `--algo_name`: 算法名称，可选 `MAPPO` 或 `Meta-MAPPO`。
- `--num_envs`: 并行环境数量，默认 `8`。
- `--max_episode_steps`: 每个回合最大步数，greedy 分支建议 `256`。
- `--max_train_steps`: 最大环境交互步数，greedy 分支建议 `256000000`。
- `--evaluate_freq`: 评估频率，单位是 `total_steps // max_episode_steps`，建议 `1000`。
- `--save_freq`: 保存频率，单位是 `total_steps // max_episode_steps`，建议 `2000`。
- `--policy_dist`: 策略分布类型，目前使用 `Gaussian`。
- `--save_dir`: 模型保存目录，默认 `./data`。
- `--model_dir`: 模型加载目录，评估时必须指向具体 checkpoint 目录。
- `--use_state_norm`: 是否使用状态归一化。
- `--use_reward_scaling`: 是否使用奖励缩放。
- `--fixed_task`: 固定训练任务，支持 T0~T5 的 task_id，即 `0~5`。

---

## 11. 项目结构

```text
MAPPO-pro/
├── algorithms/
│   ├── mappo.py                     # 基线算法 MAPPO
│   └── meta_mappo.py                # 一阶元学习 Meta-MAPPO
│
├── env/
│   ├── MPE_env.py                   # 多智能体环境封装
│   ├── environment.py               # MPE 环境包装器
│   ├── scenarios/
│   │   └── air_combat_2v2.py        # 2v2 多无人机空战环境；蓝方为协同贪婪对手
│   └── _mpe_utils/
│       ├── core.py                  # 物理状态更新与实体定义
│       ├── rendering.py             # 渲染模块
│       └── scenario.py              # 场景基类
│
├── utils/
│   ├── normalization.py             # 状态/奖励归一化
│   └── replaybuffer.py              # PPO / Meta-MAPPO 经验缓存
│
├── train_parallel.py                # 当前主训练脚本
├── run_experiments.py               # 批量训练脚本
├── evaluate.py                      # 鲁棒性、战术效能、快速适应评估脚本
├── run_fast_adaptation.py           # 批量快速适应评估脚本
├── plot_3D.py                       # 三维轨迹可视化
├── result/
│   ├── plot_combined.py             # 综合评估图绘制
│   └── plot_TensorBoard.py          # TensorBoard 曲线提取与绘制
└── README.md
```

---

## 12. 训练技巧

当前代码中常用训练技巧包括：

1. **Advantage Normalization**：优势函数归一化。
2. **State Normalization**：状态归一化。
3. **Reward Scaling**：奖励缩放。
4. **Policy Entropy**：策略熵正则化。
5. **Learning Rate Decay**：学习率衰减。
6. **Gradient Clip**：梯度裁剪。
7. **Orthogonal Initialization**：正交初始化。
8. **Adam Optimizer Epsilon Parameter**：Adam 优化器 epsilon 设置。
9. **Tanh Activation Function**：Tanh 激活函数。
10. **Active Mask**：屏蔽死亡红方智能体的无效经验。
11. **Tanh-squashed Gaussian**：连续有界动作分布，避免简单 clip 后 log_prob 不一致。
12. **Support/Query Split**：Meta-MAPPO 中用于一阶元训练的 support/query 数据划分。

---

## 13. 推荐论文实验流程

建议按照以下顺序组织实验：

1. 使用 `train_parallel.py` 或 `run_experiments.py` 训练 MAPPO 与 Meta-MAPPO。
2. 使用 TensorBoard 观察训练任务上的：
   - `Performance/Win_Rate`
   - `Performance/Full_Kill_WinRate`
   - `Performance/No_Loss_WinRate`
   - `eval/reward`
3. 使用 `evaluate.py` 在 U0~U4 未知任务上评估泛化能力。
4. 使用 `run_fast_adaptation.py` 比较快速适应前后性能。
5. 使用 `plot_combined.py` 和 `plot_TensorBoard.py` 生成论文图表。

论文中建议重点报告：

```text
Win Rate
Full-Kill Win Rate
No-Loss Win Rate
Loss-Exchange Ratio
Average Time-to-Kill
eval/reward
Fast Adaptation Improvement
```
