import csv
import json
import re
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
from env.MPE_env import MPEEnv
from env.scenarios.air_combat_2v2 import META_TRAIN_TASK_IDS
from evaluate import adapt_agent_on_task, evaluate_combat_metrics, evaluate_policy
from utils.normalization import Normalization


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
DEFAULT_MAPPO_MODEL_DIRS = (
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\750000"
)

DEFAULT_META_MAPPO_MODEL_DIRS = (
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\750000"
)


# ============================================================
# 2. 快速适应实验设置
# ============================================================
EVAL_TASKS = "6,7,8,9,10,11,12,13"      # U0~U7
ADAPT_K_LIST = [0, 5, 10, 20]
ADAPT_EPOCHS = 1
QUERY_EPISODES = 100

MAX_EPISODE_STEPS = 256
POLICY_DIST = "Gaussian"
ADAPT_REWARD_SCALE = 0.1
OUTPUT_DIR = Path("./result/fast_adaptation")
EVALUATE_ONLY_LATEST_CHECKPOINT = True


MODEL_DIRS = {
    "MAPPO": DEFAULT_MAPPO_MODEL_DIRS,
    "Meta-MAPPO": DEFAULT_META_MAPPO_MODEL_DIRS,
}


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


def parse_task_ids(task_string):
    return [int(x.strip()) for x in task_string.split(",") if x.strip()]


def extract_train_step(model_dir):
    # Use checkpoint folder names such as model/310000 as the training-step x-axis.
    match = re.search(r"(\d+)$", Path(model_dir).name)
    if match:
        return int(match.group(1))
    return None


def is_checkpoint_dir(model_dir):
    model_dir = Path(model_dir)
    return (
        (model_dir / "actor_shared.pt").exists()
        and (model_dir / "critic_shared.pt").exists()
    )


def resolve_model_dirs(model_dirs):
    # Usage example: MODEL_DIRS["MAPPO"] can be a checkpoint path, a comma-separated
    # checkpoint list, a Python list, or a parent model directory containing 100000/.
    if isinstance(model_dirs, (list, tuple)):
        raw_dirs = [str(item) for item in model_dirs]
    else:
        raw_dirs = [
            item.strip()
            for item in str(model_dirs).split(",")
            if item.strip()
        ]

    resolved = []
    for raw_dir in raw_dirs:
        path = Path(raw_dir)
        if is_checkpoint_dir(path):
            resolved.append(path)
            continue

        if path.exists():
            children = [
                child
                for child in path.iterdir()
                if child.is_dir() and child.name.isdigit() and is_checkpoint_dir(child)
            ]
            resolved.extend(sorted(children, key=lambda item: int(item.name)))
            continue

        resolved.append(path)

    return resolved


def build_eval_args(algo_name, model_dir, adapt_k):
    return Namespace(
        scenario_name="air_combat_2v2",
        algo_name=algo_name,
        date="eval",
        save_dir="./data",
        max_action=1.0,
        model_dir=str(model_dir),
        policy_dist=POLICY_DIST,
        hidden_width=256,
        max_episode_steps=MAX_EPISODE_STEPS,
        max_train_steps=256000000,
        batch_size=6000,
        mini_batch_size=1000,
        K_epochs=4,
        lr_a=3e-5,
        lr_c=3e-4,
        gamma=0.99,
        lamda=0.95,
        epsilon=0.15,
        entropy_coef=0.03,
        use_adv_norm=True,
        use_state_norm=True,
        use_lr_decay=True,
        use_grad_clip=True,
        use_orthogonal_init=True,
        set_adam_eps=True,
        use_tanh=True,
        eval_task=-1,
        eval_adaptation=True,
        only_adaptation=True,
        adapt_tasks=EVAL_TASKS,
        adapt_support_episodes=adapt_k,
        adapt_query_episodes=QUERY_EPISODES,
        adapt_epochs=ADAPT_EPOCHS,
        adapt_reward_scale=ADAPT_REWARD_SCALE,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )


def load_state_norm(args):
    if not args.use_state_norm:
        return None

    state_norm = Normalization(shape=args.state_dim)
    mean_path = Path(args.model_dir) / "norm_mean.npy"
    std_path = Path(args.model_dir) / "norm_std.npy"

    if mean_path.exists() and std_path.exists():
        state_norm.running_ms.mean = np.load(mean_path)
        state_norm.running_ms.std = np.load(std_path)
        return state_norm

    return None


def build_agent_and_state_norm(args):
    env = MPEEnv(args)
    try:
        args.state_dim = env.observation_space[0].shape[0]
        args.action_dim = env.action_space[0].shape[0]
        args.max_action = float(env.action_space[0].high[0])
        args.n_red = 2
        args.task_dim = len(META_TRAIN_TASK_IDS)
        args.share_state_dim = args.state_dim * args.n_red + args.task_dim + args.n_red

        if args.algo_name == "Meta-MAPPO":
            agent = Meta_MAPPO_Continuous(args)
        else:
            agent = MAPPO_Continuous(args)

        agent.restore(0)
        state_norm = load_state_norm(args)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    return agent, state_norm


def evaluate_agent(args, agent, state_norm, task_id):
    env = MPEEnv(args)
    agents = [agent, agent, None, None]
    try:
        metrics = evaluate_combat_metrics(
            args=args,
            env=env,
            agents=agents,
            state_norm=state_norm,
            times=args.adapt_query_episodes,
            task_id=task_id,
        )
        avg_reward = evaluate_policy(
            args=args,
            env=env,
            agents=agents,
            state_norm=state_norm,
            times=args.adapt_query_episodes,
            task_id=task_id,
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    return metrics, float(avg_reward)


def run_one_experiment(
    algo_name,
    model_dir,
    checkpoint_index,
    adapt_k,
):
    print("\n" + "=" * 80)
    print(f"开始实验: {algo_name}")
    print(f"模型路径: {model_dir}")
    print(f"K support episodes: {adapt_k}")
    print(f"Adapt epochs: {ADAPT_EPOCHS}")
    print("=" * 80)

    train_step = extract_train_step(model_dir)
    args = build_eval_args(algo_name, model_dir, adapt_k)
    task_ids = parse_task_ids(EVAL_TASKS)
    results = []

    for task_id in task_ids:
        # Reload from checkpoint per task/K to prevent adaptation contamination.
        base_agent, state_norm = build_agent_and_state_norm(args)

        if adapt_k == 0:
            eval_agent = base_agent
            support_count = 0
            adapt_status = "zero_shot"
        else:
            # adapt_agent_on_task deep-copies base_agent before PPO updates,
            # so each task and K value starts from the same checkpoint policy.
            eval_agent, support_count = adapt_agent_on_task(
                args=args,
                base_agent=base_agent,
                state_norm=state_norm,
                task_id=task_id,
            )
            adapt_status = "adapted"

        metrics, avg_reward = evaluate_agent(args, eval_agent, state_norm, task_id)
        (
            win_rate,
            exchange_ratio,
            avg_win_steps,
            avg_energy,
            total_kills,
            total_deaths,
        ) = metrics

        win_rate_percent = float(win_rate * 100.0)
        task_label = f"U{task_id - 6}"

        row = {
            "algo_name": algo_name,
            "task_id": int(task_id),
            "task_label": task_label,
            "train_step": train_step,
            "checkpoint_index": int(checkpoint_index),
            "model_dir": str(model_dir),
            "adapt_k": int(adapt_k),
            "adapt_epochs": int(ADAPT_EPOCHS),
            "support_samples": int(support_count),
            "adapt_status": adapt_status,
            "win_rate": win_rate_percent,
            "avg_reward": float(avg_reward),
            "exchange_ratio": float(exchange_ratio),
            "avg_win_steps": float(avg_win_steps),
            "avg_energy": float(avg_energy),
            "total_kills": int(total_kills),
            "total_deaths": int(total_deaths),
            "direct_win_rate": win_rate_percent if adapt_k == 0 else None,
            "adapted_win_rate": win_rate_percent if adapt_k > 0 else None,
        }
        results.append(row)

        status_text = " zero-shot" if adapt_k == 0 else ""
        print(
            f">>> {algo_name} task={task_id} step={train_step} "
            f"K={adapt_k}{status_text} | "
            f"win={row['win_rate']:.1f}% | "
            f"reward={row['avg_reward']:.2f}"
        )

    return results



def get_plot_step(item):
    if item.get("train_step") is not None:
        return item["train_step"]
    return item["checkpoint_index"]


def get_checkpoint_rank(item):
    if item.get("train_step") is not None:
        return int(item["train_step"])
    return int(item["checkpoint_index"])


def select_latest_checkpoint_results(results):
    latest_rank = {}
    for item in results:
        algo_name = item["algo_name"]
        latest_rank[algo_name] = max(
            latest_rank.get(algo_name, -1),
            get_checkpoint_rank(item),
        )

    return [
        item for item in results
        if get_checkpoint_rank(item) == latest_rank[item["algo_name"]]
    ]


def sorted_algo_names(results):
    order = {"MAPPO": 0, "Meta-MAPPO": 1}
    return sorted(
        {item["algo_name"] for item in results},
        key=lambda name: order.get(name, 99),
    )


def save_paper_table(results, output_dir):
    latest_results = select_latest_checkpoint_results(results)
    table = {}

    for algo_name in sorted_algo_names(latest_results):
        row = {}
        algo_items = [
            item for item in latest_results
            if item["algo_name"] == algo_name
        ]
        for adapt_k in ADAPT_K_LIST:
            values = [
                float(item["win_rate"])
                for item in algo_items
                if int(item["adapt_k"]) == adapt_k
            ]
            row[f"P{adapt_k}win"] = round(float(np.mean(values)), 2) if values else None
        table[algo_name] = row

    csv_path = output_dir / "table_5_6_few_shot_winrate.csv"
    json_path = output_dir / "table_5_6_few_shot_winrate.json"
    headers = ["algo_name", "P0win", "P5win", "P10win", "P20win"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for algo_name in sorted_algo_names(latest_results):
            writer.writerow({"algo_name": algo_name, **table[algo_name]})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=4)

    print(csv_path)
    print(json_path)


def plot_few_shot_winrate_curve(results, output_dir):
    latest_results = select_latest_checkpoint_results(results)
    sns.set_theme(style="darkgrid")
    colors = {
        "MAPPO": "#4C72B0",
        "Meta-MAPPO": "#DD8452",
    }

    plt.figure(figsize=(10, 6))
    for algo_name in sorted_algo_names(latest_results):
        means = []
        stds = []
        for adapt_k in ADAPT_K_LIST:
            values = [
                float(item["win_rate"])
                for item in latest_results
                if item["algo_name"] == algo_name and int(item["adapt_k"]) == adapt_k
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            stds.append(float(np.std(values)) if values else 0.0)

        x_values = np.array(ADAPT_K_LIST)
        y_values = np.array(means, dtype=float)
        std_values = np.array(stds, dtype=float)
        lower = np.clip(y_values - std_values, 0.0, 100.0)
        upper = np.clip(y_values + std_values, 0.0, 100.0)
        color = colors.get(algo_name, "#000000")

        plt.fill_between(x_values, lower, upper, alpha=0.2, color=color)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=7,
            linewidth=2.5,
            color=color,
            label=algo_name,
        )

    plt.xlabel("Adaptation Episodes K", fontsize=14, fontweight="bold")
    plt.ylabel("Mean win rate (%)", fontsize=14, fontweight="bold")
    plt.title("Few-shot Adaptation Win Rate on Unseen Tasks", fontsize=16, fontweight="bold")
    plt.xticks(ADAPT_K_LIST)
    plt.ylim(0, 100)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    curve_path = output_dir / "few_shot_adaptation_winrate_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(curve_path)


def plot_few_shot_task_curves(results, output_dir):
    latest_results = select_latest_checkpoint_results(results)
    sns.set_theme(style="darkgrid")
    colors = {
        "MAPPO": "#4C72B0",
        "Meta-MAPPO": "#DD8452",
    }

    task_ids = sorted({int(item["task_id"]) for item in latest_results})
    for task_id in task_ids:
        task_label = f"U{task_id - 6}"
        plt.figure(figsize=(10, 6))

        for algo_name in sorted_algo_names(latest_results):
            y_values = []
            for adapt_k in ADAPT_K_LIST:
                values = [
                    float(item["win_rate"])
                    for item in latest_results
                    if item["algo_name"] == algo_name
                    and int(item["task_id"]) == task_id
                    and int(item["adapt_k"]) == adapt_k
                ]
                y_values.append(float(np.mean(values)) if values else np.nan)

            plt.plot(
                ADAPT_K_LIST,
                y_values,
                marker="o",
                markersize=7,
                linewidth=2.5,
                color=colors.get(algo_name, "#000000"),
                label=algo_name,
            )

        plt.xlabel("Adaptation Episodes K", fontsize=14, fontweight="bold")
        plt.ylabel("Win rate (%)", fontsize=14, fontweight="bold")
        plt.title(f"Few-shot Adaptation Win Rate on {task_label}", fontsize=16, fontweight="bold")
        plt.xticks(ADAPT_K_LIST)
        plt.ylim(0, 100)
        plt.legend(fontsize=12, loc="best")
        plt.tight_layout()

        curve_path = output_dir / f"few_shot_adaptation_winrate_{task_label}.png"
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(curve_path)


def save_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "fast_adaptation_summary.json"
    csv_path = output_dir / "fast_adaptation_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for item in results:
                writer.writerow(item)

    print("\nFast adaptation results:")
    print(json_path)
    print(csv_path)

    print("\nPaper table:")
    save_paper_table(results, output_dir)

    print("\nFew-shot adaptation images:")
    plot_few_shot_winrate_curve(results, output_dir)
    plot_few_shot_task_curves(results, output_dir)


def main():
    print("Start fast-adaptation evaluation.")
    all_results = []

    for algo_name in RUN_ALGOS:
        if algo_name not in MODEL_DIRS:
            raise KeyError(f"MODEL_DIRS missing model path for {algo_name}")

        model_dirs = resolve_model_dirs(MODEL_DIRS[algo_name])
        if EVALUATE_ONLY_LATEST_CHECKPOINT and model_dirs:
            model_dirs = [model_dirs[-1]]

        for checkpoint_index, model_dir in enumerate(model_dirs):
            model_dir = check_model_dir(model_dir)
            for adapt_k in ADAPT_K_LIST:
                results = run_one_experiment(
                    algo_name=algo_name,
                    model_dir=model_dir,
                    checkpoint_index=checkpoint_index,
                    adapt_k=adapt_k,
                )
                all_results.extend(results)

    save_results(all_results, OUTPUT_DIR)
    print("\nAll fast-adaptation experiments finished.")


if __name__ == "__main__":
    main()
