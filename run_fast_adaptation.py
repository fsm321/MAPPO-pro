import subprocess
import sys
from pathlib import Path


# ============================================================
# 1. 在这里填写你的模型路径
# ============================================================
# 路径应该指向具体 checkpoint 文件夹，例如：
# ./data/Meta-MAPPO_seed10_xxxx/model/100000
# 里面应该有：
# actor_shared.pt
# critic_shared.pt
# norm_mean.npy
# norm_std.npy
MODEL_DIRS = {
    "MAPPO": r"./data/你的MAPPO目录/model/检查点",
    "Meta-MAPPO": r"./data/你的Meta-MAPPO目录/model/检查点",
}


# ============================================================
# 2. 快速适应实验设置
# ============================================================
EVAL_TASKS = "6,7,8,9,10"      # U0~U4
SUPPORT_EPISODES_LIST = [1, 3, 5]
ADAPT_EPOCHS_LIST = [1, 3]
QUERY_EPISODES = 100

MAX_EPISODE_STEPS = 500
POLICY_DIST = "Gaussian"


# ============================================================
# 3. 是否只评估 Meta-MAPPO
# ============================================================
# 如果你只想评估 Meta-MAPPO，把 RUN_ALGOS 改成 ["Meta-MAPPO"]
# 如果想同时对比 MAPPO 和 Meta-MAPPO，就保留两个。
RUN_ALGOS = ["MAPPO", "Meta-MAPPO"]


def check_model_dir(model_dir):
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    actor_path = model_dir / "actor_shared.pt"
    critic_path = model_dir / "critic_shared.pt"

    if not actor_path.exists():
        raise FileNotFoundError(f"缺少 actor_shared.pt: {actor_path}")

    if not critic_path.exists():
        raise FileNotFoundError(f"缺少 critic_shared.pt: {critic_path}")

    return str(model_dir)


def run_one_experiment(algo_name, model_dir, support_episodes, adapt_epochs):
    print("\n" + "=" * 80)
    print(f"开始实验: {algo_name}")
    print(f"模型路径: {model_dir}")
    print(f"Support episodes: {support_episodes}")
    print(f"Adapt epochs: {adapt_epochs}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "evaluate.py",
        "--algo_name", algo_name,
        "--model_dir", model_dir,
        "--policy_dist", POLICY_DIST,
        "--max_episode_steps", str(MAX_EPISODE_STEPS),
        "--eval_adaptation",
        "--only_adaptation",
        "--adapt_tasks", EVAL_TASKS,
        "--adapt_support_episodes", str(support_episodes),
        "--adapt_epochs", str(adapt_epochs),
        "--adapt_query_episodes", str(QUERY_EPISODES),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"实验失败: {algo_name}, support={support_episodes}, epochs={adapt_epochs}")
    else:
        print(f"实验完成: {algo_name}, support={support_episodes}, epochs={adapt_epochs}")


def main():
    print("准备开始快速适应评估实验")

    for algo_name in RUN_ALGOS:
        if algo_name not in MODEL_DIRS:
            raise KeyError(f"MODEL_DIRS 中没有配置 {algo_name} 的模型路径")

        model_dir = check_model_dir(MODEL_DIRS[algo_name])

        for support_episodes in SUPPORT_EPISODES_LIST:
            for adapt_epochs in ADAPT_EPOCHS_LIST:
                run_one_experiment(
                    algo_name=algo_name,
                    model_dir=model_dir,
                    support_episodes=support_episodes,
                    adapt_epochs=adapt_epochs,
                )

    print("\n所有快速适应实验运行完毕")


if __name__ == "__main__":
    main()
