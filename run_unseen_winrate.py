import argparse
import copy
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from env.MPE_env import MPEEnv
from env.scenarios.air_combat_2v2 import META_TRAIN_TASK_IDS
from utils.normalization import Normalization
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous


TASKS = {
    6: "U0_mixed_flank_then_decoy",
    7: "U1_stronger_dynamics",
    8: "U2_large_initial_perturbation",
    9: "U3_obs_noise_red_failure",
    10: "U4_weapon_parameter_shift",
    11: "U5_head_on_neutral",
    12: "U6_disadvantage_tail_chase",
    13: "U7_spatial_constraint",
}


# ============================================================
# Checkpoint path config
#   "./data/MAPPO_xxx/model/100000,./data/MAPPO_xxx/model/200000"
# ============================================================
DEFAULT_MAPPO_MODEL_DIRS = (
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\50000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\100000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\150000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\200000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\250000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\300000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\350000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\400000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\450000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\500000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\550000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\600000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\650000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\700000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2\model\750000"
)

DEFAULT_META_MAPPO_MODEL_DIRS = (
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\50000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\100000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\150000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\200000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\250000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\300000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\350000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\400000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\450000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\500000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\550000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\600000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\650000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\700000,"
r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005\model\750000"
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")


def normalize_obs(args, obs, state_norm):
    if args.use_state_norm and state_norm is not None:
        return state_norm(obs, update=False)
    return obs


def load_state_norm(args):
    if not args.use_state_norm:
        return None

    state_norm = Normalization(shape=args.state_dim)
    mean_path = Path(args.model_dir) / "norm_mean.npy"
    std_path = Path(args.model_dir) / "norm_std.npy"

    if mean_path.exists() and std_path.exists():
        state_norm.running_ms.mean = np.load(mean_path)
        state_norm.running_ms.std = np.load(std_path)
        print("状态归一化统计量加载成功。")
        return state_norm

    print("未找到 norm_mean.npy / norm_std.npy，将使用原始状态评估。")
    return None


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


def parse_model_dirs(model_dir_string):
    # Allow comma-separated checkpoints so plots can use training step as x-axis.
    return [
        item.strip()
        for item in str(model_dir_string).split(",")
        if item.strip()
    ]


def extract_train_step(model_dir):
    # Read the checkpoint step from common paths such as model/310000.
    match = re.search(r"(\d+)$", Path(model_dir).name)
    if match:
        return int(match.group(1))
    return None


def build_agent(args, env):
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
    return agent


def evaluate_win_rate(args, agent, state_norm, task_id):
    env = MPEEnv(args)
    total_wins = 0
    episode_lengths = []
    episode_rewards = []

    try:
        for ep in range(args.eval_episodes):
            s = env.reset(task_id=task_id)
            dones = np.zeros(env.n, dtype=bool)
            episode_steps = 0
            episode_reward = 0.0

            while (not np.all(dones)) and episode_steps < args.max_episode_steps:
                episode_steps += 1
                actions = []

                for agent_id in range(env.n):
                    if agent_id < 2:
                        obs = normalize_obs(args, s[agent_id], state_norm)
                        a = agent.choose_action_deterministic(obs)

                        if args.policy_dist == "Beta":
                            action = 2 * (a - 0.5) * args.max_action
                        else:
                            action = a

                        actions.append(action)
                    else:
                        actions.append(np.zeros(args.action_dim))

                s_next, r, done, info = env.step(actions)
                episode_reward += float(np.sum(r[:2]))
                s = s_next
                dones = np.asarray(done, dtype=bool)

            world = (
                getattr(env, "world", None)
                or getattr(getattr(env, "env", None), "world", None)
                or getattr(getattr(env, "unwrapped", None), "world", None)
            )

            if world is not None:
                red_dead = sum(
                    1 for a in world.agents
                    if a.team == 0 and getattr(a, "is_dead", False)
                )
                blue_dead = sum(
                    1 for a in world.agents
                    if a.team == 1 and getattr(a, "is_dead", False)
                )
            else:
                red_dead = int(np.sum(dones[:2]))
                blue_dead = int(np.sum(dones[2:]))

            # 与 evaluate.py 中的胜利判定保持一致：
            # 蓝方全灭，或蓝方损失数大于红方损失数，记为红方胜利。
            is_win = (blue_dead == 2) or (blue_dead > red_dead)

            if is_win:
                total_wins += 1

            episode_lengths.append(episode_steps)
            episode_rewards.append(episode_reward)

            if (ep + 1) % 10 == 0:
                curr_win_rate = total_wins / (ep + 1) * 100.0
                curr_reward = float(np.mean(episode_rewards))
                print(f"  Current average reward: {curr_reward:.2f}")
                print(
                    f"  已评估 {ep + 1}/{args.eval_episodes} 局，"
                    f"当前胜率: {curr_win_rate:.1f}%"
                )

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    win_rate = total_wins / args.eval_episodes * 100.0
    avg_steps = float(np.mean(episode_lengths)) if episode_lengths else args.max_episode_steps
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

    return {
        "algo_name": args.algo_name,
        "task_id": int(task_id),
        "task_name": TASKS[task_id],
        "train_step": getattr(args, "train_step", None),
        "checkpoint_index": int(getattr(args, "checkpoint_index", 0)),
        "eval_episodes": int(args.eval_episodes),
        "win_count": int(total_wins),
        "win_rate": float(win_rate),
        "avg_reward": float(avg_reward),
        "avg_episode_steps": float(avg_steps),
        "model_dir": str(args.model_dir),
    }


def evaluate_algo_on_tasks(
    base_args,
    algo_name,
    model_dir,
    task_ids,
    checkpoint_index=0,
    train_step=None,
):
    args = copy.deepcopy(base_args)
    args.algo_name = algo_name
    args.model_dir = model_dir
    args.checkpoint_index = checkpoint_index
    args.train_step = train_step

    print("\n" + "=" * 100)
    print(f"开始评估算法: {algo_name}")
    print(f"模型路径: {model_dir}")
    print(f"Checkpoint step: {train_step if train_step is not None else checkpoint_index}")
    print("=" * 100)

    check_model_dir(model_dir)

    env = MPEEnv(args)
    try:
        agent = build_agent(args, env)
        state_norm = load_state_norm(args)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    results = []

    for task_id in task_ids:
        print("\n" + "-" * 80)
        print(f"评估任务: {TASKS[task_id]} | task_id={task_id}")
        print("-" * 80)

        result = evaluate_win_rate(args, agent, state_norm, task_id)
        results.append(result)

        print(
            f">>> {algo_name} | {TASKS[task_id]} | "
            f"胜率: {result['win_rate']:.1f}% "
            f"({result['win_count']}/{result['eval_episodes']}) | "
            f"平均步数: {result['avg_episode_steps']:.1f}"
        )

    return results


def get_plot_step(item):
    if item.get("train_step") is not None:
        return item["train_step"]
    return item["checkpoint_index"]


def plot_summary_curve(results, output_dir, metric_name, ylabel, filename, title):
    # Match result/plot_TensorBoard.py and show checkpoint points explicitly.
    if not results:
        return

    sns.set_theme(style="darkgrid")
    colors = {
        "Meta-MAPPO": "#DD8452",
        "MAPPO": "#4C72B0",
    }

    grouped = {}
    for item in results:
        key = (item["algo_name"], get_plot_step(item))
        grouped.setdefault(key, []).append(float(item[metric_name]))

    series = {}
    for (algo_name, plot_step), values in grouped.items():
        series.setdefault(algo_name, []).append((
            plot_step,
            float(np.mean(values)),
            float(np.std(values)),
        ))

    plt.figure(figsize=(10, 6))
    for algo_name, points in sorted(series.items()):
        points = sorted(points, key=lambda x: x[0])
        x_values = np.array([x for x, _, _ in points])
        y_values = np.array([y for _, y, _ in points])
        std_values = np.array([std for _, _, std in points])
        color = colors.get(algo_name, "#000000")

        lower = y_values - std_values
        upper = y_values + std_values
        if metric_name == "win_rate":
            lower = np.clip(lower, 0.0, 100.0)
            upper = np.clip(upper, 0.0, 100.0)

        if len(points) > 1:
            plt.fill_between(x_values, lower, upper, alpha=0.2, color=color)

        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=6,
            label=algo_name,
            linewidth=1.5,
            color=color,
        )

    plt.xlabel("Step", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    if metric_name == "win_rate":
        plt.ylim(0, 100)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    curve_path = output_dir / filename
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(curve_path)


def plot_task_points(results, output_dir, metric_name, ylabel, filename, title):

    if not results:
        return

    sns.set_theme(style="darkgrid")
    colors = {
        "Meta-MAPPO": "#DD8452",
        "MAPPO": "#4C72B0",
    }

    grouped = {}
    for item in results:
        key = (item["algo_name"], item["task_id"])
        grouped.setdefault(key, []).append(float(item[metric_name]))

    series = {}
    for (algo_name, task_id), values in grouped.items():
        series.setdefault(algo_name, []).append((task_id, float(np.mean(values))))

    plt.figure(figsize=(10, 6))
    for algo_name, points in sorted(series.items()):
        points = sorted(points, key=lambda x: x[0])
        x_values = np.array([x for x, _ in points])
        y_values = np.array([y for _, y in points])
        labels = [TASKS.get(int(x), f"Task_{int(x)}").split("_", 1)[0] for x in x_values]
        color = colors.get(algo_name, "#000000")

        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=6,
            label=algo_name,
            linewidth=1.5,
            color=color,
        )

    plt.xlabel("Meta-test task", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(x_values, labels)
    if metric_name == "win_rate":
        plt.ylim(0, 100)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    curve_path = output_dir / filename
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(curve_path)


def plot_task_step_curves(results, output_dir, metric_name, ylabel, filename_prefix, title_prefix):
    # For each meta-test task, plot metric versus checkpoint step. Each figure
    # compares MAPPO and Meta-MAPPO as two curves across multiple checkpoints.
    if not results:
        return

    sns.set_theme(style="darkgrid")
    colors = {
        "Meta-MAPPO": "#DD8452",
        "MAPPO": "#4C72B0",
    }

    task_ids = sorted({int(item["task_id"]) for item in results})
    for task_id in task_ids:
        task_results = [item for item in results if int(item["task_id"]) == task_id]
        series = {}

        for item in task_results:
            algo_name = item["algo_name"]
            plot_step = get_plot_step(item)
            metric_value = float(item[metric_name])
            series.setdefault(algo_name, []).append((plot_step, metric_value))

        plt.figure(figsize=(10, 6))
        for algo_name, points in sorted(series.items()):
            points = sorted(points, key=lambda x: x[0])
            x_values = np.array([x for x, _ in points])
            y_values = np.array([y for _, y in points])
            color = colors.get(algo_name, "#000000")

            plt.plot(
                x_values,
                y_values,
                marker="o",
                markersize=3,
                label=algo_name,
                linewidth=2.5,
                color=color,
            )

        task_name = TASKS.get(task_id, f"Task_{task_id}")
        task_label = task_name.split("_", 1)[0]
        plt.xlabel("Step", fontsize=14, fontweight="bold")
        plt.ylabel(ylabel, fontsize=14, fontweight="bold")
        plt.title(f"{title_prefix} {task_label}", fontsize=16, fontweight="bold")
        if metric_name == "win_rate":
            plt.ylim(0, 100)
        plt.legend(fontsize=12, loc="best")
        plt.tight_layout()

        curve_path = output_dir / f"{filename_prefix}_{task_label}.png"
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(curve_path)


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "unseen_winrate_summary.json"
    csv_path = output_dir / "unseen_winrate_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    headers = [
        "algo_name",
        "task_id",
        "task_name",
        "train_step",
        "checkpoint_index",
        "eval_episodes",
        "win_count",
        "win_rate",
        "avg_reward",
        "avg_episode_steps",
        "model_dir",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in results:
            writer.writerow(item)

    print("\n结果已保存：")
    print(json_path)
    print(csv_path)
    print("\nCurve images:")
    plot_task_step_curves(
        results,
        output_dir,
        metric_name="win_rate",
        ylabel="Win rate (%)",
        filename_prefix="unseen_winrate_step_curve",
        title_prefix="Unseen Win Rate",
    )
    plot_task_step_curves(
        results,
        output_dir,
        metric_name="avg_reward",
        ylabel="Reward",
        filename_prefix="unseen_reward_step_curve",
        title_prefix="Unseen Reward",
    )
    plot_task_points(
        results,
        output_dir,
        metric_name="win_rate",
        ylabel="Win rate (%)",
        filename="unseen_winrate_by_task.png",
        title="Unseen Task Win Rate by Test Point",
    )
    plot_task_points(
        results,
        output_dir,
        metric_name="avg_reward",
        ylabel="Reward",
        filename="unseen_reward_by_task.png",
        title="Unseen Task Reward by Test Point",
    )
    plot_summary_curve(
        results,
        output_dir,
        metric_name="win_rate",
        ylabel="Mean win rate (%)",
        filename="unseen_winrate_curve.png",
        title="Unseen Task Win Rate",
    )
    plot_summary_curve(
        results,
        output_dir,
        metric_name="avg_reward",
        ylabel="Mean reward",
        filename="unseen_reward_curve.png",
        title="Unseen Task Reward",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--date", type=str, default="eval")
    parser.add_argument("--mappo_model_dir", type=str, default=DEFAULT_MAPPO_MODEL_DIRS)
    parser.add_argument("--meta_mappo_model_dir", type=str, default=DEFAULT_META_MAPPO_MODEL_DIRS)

    parser.add_argument("--algos", type=str, default="MAPPO,Meta-MAPPO")
    parser.add_argument("--tasks", type=str, default="6,7,8,9,10,11,12,13")

    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=256)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=256)

    parser.add_argument("--use_state_norm", type=str2bool, default=True)
    parser.add_argument("--use_tanh", type=str2bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True)
    parser.add_argument("--set_adam_eps", type=str2bool, default=True)
    parser.add_argument("--use_grad_clip", type=str2bool, default=True)
    parser.add_argument("--use_lr_decay", type=str2bool, default=True)
    parser.add_argument("--use_adv_norm", type=str2bool, default=True)

    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--mini_batch_size", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=256000000)
    parser.add_argument("--lr_a", type=float, default=3e-5)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.03)

    parser.add_argument("--output_dir", type=str, default="./result/unseen_winrate")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    algos = [x.strip() for x in args.algos.split(",") if x.strip()]
    task_ids = [int(x.strip()) for x in args.tasks.split(",") if x.strip()]

    for task_id in task_ids:
        if task_id not in TASKS:
            raise ValueError(f"不支持的任务 task_id={task_id}，应为 6~13。")

    all_results = []

    for algo_name in algos:
        if algo_name == "MAPPO":
            if not args.mappo_model_dir:
                raise ValueError("需要提供 --mappo_model_dir")
            for checkpoint_index, model_dir in enumerate(parse_model_dirs(args.mappo_model_dir)):
                results = evaluate_algo_on_tasks(
                    base_args=args,
                    algo_name="MAPPO",
                    model_dir=model_dir,
                    task_ids=task_ids,
                    checkpoint_index=checkpoint_index,
                    train_step=extract_train_step(model_dir),
                )
                all_results.extend(results)

        elif algo_name == "Meta-MAPPO":
            if not args.meta_mappo_model_dir:
                raise ValueError("需要提供 --meta_mappo_model_dir")
            for checkpoint_index, model_dir in enumerate(parse_model_dirs(args.meta_mappo_model_dir)):
                results = evaluate_algo_on_tasks(
                    base_args=args,
                    algo_name="Meta-MAPPO",
                    model_dir=model_dir,
                    task_ids=task_ids,
                    checkpoint_index=checkpoint_index,
                    train_step=extract_train_step(model_dir),
                )
                all_results.extend(results)

        else:
            raise ValueError(f"不支持的算法: {algo_name}")

    save_results(all_results, args.output_dir)

    print("\nU0~U7 未见任务胜率评估完成。")


if __name__ == "__main__":
    main()
