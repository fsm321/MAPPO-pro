import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from env.MPE_env import MPEEnv
from utils.normalization import Normalization
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous


TASKS = {
    0: "T0_head_on_chase",
    1: "T1_flank_and_press",
    2: "T2_high_low_layer",
    3: "T3_decoy_and_attack",
    4: "T4_noisy_chase",
    5: "T5_defensive_counter",
    6: "U0_mixed_flank_then_decoy",
    7: "U1_stronger_dynamics",
    8: "U2_large_initial_perturbation",
    9: "U3_obs_noise_red_failure",
    10: "U4_weapon_parameter_shift",
}

AGENT_KEYS = ["R1", "R2", "B1", "B2"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")


def get_world(env):
    return (
        getattr(env, "world", None)
        or getattr(getattr(env, "env", None), "world", None)
        or getattr(getattr(env, "unwrapped", None), "world", None)
    )


def get_agent_xyz(agent):
    return np.array([
        float(agent.state.p_pos[0]),
        float(agent.state.p_pos[1]),
        float(agent.state.z_pos),
    ], dtype=np.float32)


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

    print("未找到 norm_mean.npy / norm_std.npy，将使用原始状态绘图。")
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


def build_agent(args, env):
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])
    args.n_red = 2
    args.task_dim = 6
    args.share_state_dim = args.state_dim * args.n_red + args.task_dim + args.n_red

    if args.algo_name == "Meta-MAPPO":
        agent = Meta_MAPPO_Continuous(args)
    elif args.algo_name == "MAPPO":
        agent = MAPPO_Continuous(args)
    else:
        raise ValueError(f"不支持的算法: {args.algo_name}")

    agent.restore(0)
    return agent


def choose_red_action(args, agent, obs):
    obs = np.asarray(obs, dtype=np.float32)

    if args.deterministic:
        action = agent.choose_action_deterministic(obs)
    else:
        action, _ = agent.choose_action(obs)

    if args.policy_dist == "Beta":
        action = 2.0 * (action - 0.5) * args.max_action

    return np.asarray(action, dtype=np.float32)


def init_record(task_id, episode_id):
    record = {
        "task_id": int(task_id),
        "task_name": TASKS.get(int(task_id), f"task_{task_id}"),
        "episode_id": int(episode_id),
        "R1": [],
        "R2": [],
        "B1": [],
        "B2": [],
        "hp": {k: [] for k in AGENT_KEYS},
        "dead": {k: [] for k in AGENT_KEYS},
        "events": [],
        "failure_step": None,
        "failure_agent": None,
        "episode_steps": 0,
        "red_dead": 0,
        "blue_dead": 0,
        "red_alive": 0,
        "blue_alive": 0,
        "win": False,
        "full_kill_win": False,
        "pincer_score": 0.0,
        "failure_recovery_score": 0.0,
    }
    return record


def record_world_state(world, record):
    for i, key in enumerate(AGENT_KEYS):
        agent = world.agents[i]
        pos = get_agent_xyz(agent)

        record[key].append(pos.tolist())
        record["hp"][key].append(float(getattr(agent, "hp", 100.0)))
        record["dead"][key].append(bool(getattr(agent, "is_dead", False)))


def collect_new_death_events(world, record, prev_dead, step):
    for i, key in enumerate(AGENT_KEYS):
        agent = world.agents[i]
        now_dead = bool(getattr(agent, "is_dead", False))

        if now_dead and not prev_dead[i]:
            record["events"].append({
                "type": "destroyed",
                "step": int(step),
                "agent_id": int(i),
                "agent_key": key,
                "team": "red" if i < 2 else "blue",
                "pos": get_agent_xyz(agent).tolist(),
            })

        prev_dead[i] = now_dead


def summarize_record(world, record, step):
    red_dead = sum(
        1 for a in world.agents
        if a.team == 0 and bool(getattr(a, "is_dead", False))
    )
    blue_dead = sum(
        1 for a in world.agents
        if a.team == 1 and bool(getattr(a, "is_dead", False))
    )

    red_alive = sum(
        1 for a in world.agents
        if a.team == 0 and not bool(getattr(a, "is_dead", False))
    )
    blue_alive = sum(
        1 for a in world.agents
        if a.team == 1 and not bool(getattr(a, "is_dead", False))
    )

    record["episode_steps"] = int(step)
    record["red_dead"] = int(red_dead)
    record["blue_dead"] = int(blue_dead)
    record["red_alive"] = int(red_alive)
    record["blue_alive"] = int(blue_alive)

    record["win"] = bool((blue_dead == 2) or (blue_dead > red_dead))
    record["full_kill_win"] = bool(blue_dead == 2 and red_alive > 0)


def rollout_episode(args, agent, state_norm, task_id, episode_id, mode):
    env = MPEEnv(args)
    record = init_record(task_id, episode_id)

    try:
        state = env.reset(task_id=task_id)
        world = get_world(env)
        if world is None:
            raise RuntimeError("无法获取 env.world，请检查环境封装。")

        prev_dead = [
            bool(getattr(a, "is_dead", False))
            for a in world.agents[:4]
        ]

        record_world_state(world, record)

        for step in range(1, args.max_episode_steps + 1):
            actions = []

            for agent_id in range(env.n):
                if agent_id < 2:
                    if (
                        mode == "failure"
                        and args.force_red0_failure
                        and agent_id == 0
                        and step >= args.failure_step
                    ):
                        action = np.zeros(args.action_dim, dtype=np.float32)

                        if record["failure_step"] is None:
                            record["failure_step"] = int(step)
                            record["failure_agent"] = "R1"
                    else:
                        obs = normalize_obs(args, state[agent_id], state_norm)
                        action = choose_red_action(args, agent, obs)

                    actions.append(action)
                else:
                    actions.append(np.zeros(args.action_dim, dtype=np.float32))

            next_state, reward, done, info = env.step(actions)
            state = next_state

            world = get_world(env)
            record_world_state(world, record)
            collect_new_death_events(world, record, prev_dead, step)

            if np.all(done):
                summarize_record(world, record, step)
                break

            if step == args.max_episode_steps:
                summarize_record(world, record, step)

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    record["pincer_score"] = float(compute_pincer_score(record, args.pincer_distance, args.pincer_angle))
    record["failure_recovery_score"] = float(
        compute_failure_recovery_score(record, failure_step=args.failure_step)
    )

    return record


def collect_episodes(args, agent, state_norm, task_id, mode, num_episodes):
    records = []

    for ep in range(num_episodes):
        print(f"采样 {TASKS.get(task_id, task_id)} | {mode} | episode {ep + 1}/{num_episodes}")
        record = rollout_episode(
            args=args,
            agent=agent,
            state_norm=state_norm,
            task_id=task_id,
            episode_id=ep,
            mode=mode,
        )
        records.append(record)

        if (ep + 1) % 10 == 0:
            wins = sum(1 for r in records if r["win"])
            print(f"  当前胜率: {wins / len(records) * 100.0:.1f}% ({wins}/{len(records)})")

    return records


def angle_deg(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def compute_pincer_score(record, dist_th=5.0, angle_th=60.0):
    if len(record["R1"]) == 0:
        return 0.0

    score = 0.0
    T = min(
        len(record["R1"]),
        len(record["R2"]),
        len(record["B1"]),
        len(record["B2"]),
    )

    for t in range(T):
        r1 = np.asarray(record["R1"][t], dtype=np.float32)
        r2 = np.asarray(record["R2"][t], dtype=np.float32)

        for blue_key in ["B1", "B2"]:
            blue_id = 2 if blue_key == "B1" else 3

            if record["dead"][blue_key][t]:
                continue

            b = np.asarray(record[blue_key][t], dtype=np.float32)

            v1 = r1 - b
            v2 = r2 - b
            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            ang = angle_deg(v1, v2)

            if d1 < dist_th and d2 < dist_th and ang > angle_th:
                score += 1.0 + ang / 180.0

                blue_destroyed_later = any(
                    e["agent_id"] == blue_id and e["step"] >= t
                    for e in record["events"]
                    if e["type"] == "destroyed"
                )
                if blue_destroyed_later:
                    score += 0.5

    return float(score)


def compute_failure_recovery_score(record, failure_step=50):
    T = len(record["R2"])

    if T <= failure_step + 5:
        return -1e9

    score = 0.0

    for t in range(failure_step, T):
        r2 = np.asarray(record["R2"][t], dtype=np.float32)

        live_blue_positions = []
        for blue_key in ["B1", "B2"]:
            if t < len(record["dead"][blue_key]) and not record["dead"][blue_key][t]:
                live_blue_positions.append(np.asarray(record[blue_key][t], dtype=np.float32))

        if not live_blue_positions:
            score += 10.0
            continue

        min_d = min(np.linalg.norm(r2 - b) for b in live_blue_positions)
        score += -float(min_d)

    blue_kills_after_failure = sum(
        1 for e in record["events"]
        if e["type"] == "destroyed"
        and e["agent_id"] in [2, 3]
        and e["step"] >= failure_step
    )

    score += 30.0 * blue_kills_after_failure

    if record["win"]:
        score += 50.0

    if record["full_kill_win"]:
        score += 30.0

    return float(score)


def select_typical_win(records):
    win_records = [r for r in records if r["win"]]

    if not win_records:
        print("警告：未找到获胜回合，将选择蓝方损失最多的回合。")
        return max(records, key=lambda r: (r["blue_dead"], -r["red_dead"], -r["episode_steps"]))

    avg_steps = np.mean([r["episode_steps"] for r in win_records])

    best = min(
        win_records,
        key=lambda r: (
            abs(r["episode_steps"] - avg_steps),
            -r["blue_dead"],
            r["red_dead"],
        )
    )

    return best


def select_best_pincer(records):
    candidates = [r for r in records if r["win"] and r["pincer_score"] > 0.0]

    if not candidates:
        print("警告：未找到明显包夹获胜回合，将在所有回合中选择包夹评分最高者。")
        candidates = records

    best = max(
        candidates,
        key=lambda r: (
            r["pincer_score"],
            r["blue_dead"],
            -r["red_dead"],
            -r["episode_steps"],
        )
    )

    return best


def select_best_failure_recovery(records):
    candidates = [
        r for r in records
        if r["episode_steps"] > 50
    ]

    if not candidates:
        print("警告：未找到超过失效步长的回合，将选择所有回合中恢复评分最高者。")
        candidates = records

    best = max(
        candidates,
        key=lambda r: (
            r["failure_recovery_score"],
            r["blue_dead"],
            -r["red_dead"],
            r["win"],
        )
    )

    return best


def array_from_record(record, key):
    arr = np.asarray(record[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return arr


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2.0, x_middle + max_range / 2.0])
    ax.set_ylim3d([y_middle - max_range / 2.0, y_middle + max_range / 2.0])
    ax.set_zlim3d([max(0.0, z_middle - max_range / 2.0), z_middle + max_range / 2.0])


def plot_episode_3d(record, save_path, title, args):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {
        "R1": "red",
        "R2": "darkred",
        "B1": "blue",
        "B2": "navy",
    }

    linestyle_map = {
        "R1": "-",
        "R2": "--",
        "B1": "-",
        "B2": "--",
    }

    label_map = {
        "R1": "Red UAV 1",
        "R2": "Red UAV 2",
        "B1": "Blue UAV 1",
        "B2": "Blue UAV 2",
    }

    all_points = []

    for key in AGENT_KEYS:
        arr = array_from_record(record, key)

        if len(arr) == 0:
            continue

        all_points.append(arr)

        ax.plot(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            color=color_map[key],
            linestyle=linestyle_map[key],
            linewidth=2.2,
            label=label_map[key],
        )

        ax.scatter(
            arr[0, 0],
            arr[0, 1],
            arr[0, 2],
            color=color_map[key],
            marker="o",
            s=60,
            edgecolors="black",
            linewidths=0.6,
        )

        ax.text(
            arr[0, 0],
            arr[0, 1],
            arr[0, 2],
            f" {key}-Start",
            fontsize=8,
        )

        ax.scatter(
            arr[-1, 0],
            arr[-1, 1],
            arr[-1, 2],
            color=color_map[key],
            marker="s",
            s=60,
            edgecolors="black",
            linewidths=0.6,
        )

        ax.text(
            arr[-1, 0],
            arr[-1, 1],
            arr[-1, 2],
            f" {key}-End",
            fontsize=8,
        )

    for event in record["events"]:
        if event["type"] != "destroyed":
            continue

        pos = np.asarray(event["pos"], dtype=np.float32)
        agent_key = event["agent_key"]

        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            color="black",
            marker="x",
            s=120,
            linewidths=2.0,
        )

        ax.text(
            pos[0],
            pos[1],
            pos[2],
            f" {agent_key} destroyed@{event['step']}",
            fontsize=9,
            color="black",
        )

    if record.get("failure_step") is not None and record.get("failure_agent") is not None:
        failure_agent = record["failure_agent"]
        failure_step = int(record["failure_step"])
        arr = array_from_record(record, failure_agent)

        if len(arr) > failure_step:
            p = arr[failure_step]
            ax.scatter(
                p[0],
                p[1],
                p[2],
                color="purple",
                marker="^",
                s=100,
                edgecolors="black",
                linewidths=0.8,
            )
            ax.text(
                p[0],
                p[1],
                p[2],
                f" {failure_agent} failure@{failure_step}",
                fontsize=9,
                color="purple",
            )

    if all_points:
        stacked = np.vstack(all_points)

        margin = 0.8
        ax.set_xlim(float(stacked[:, 0].min() - margin), float(stacked[:, 0].max() + margin))
        ax.set_ylim(float(stacked[:, 1].min() - margin), float(stacked[:, 1].max() + margin))
        ax.set_zlim(max(0.0, float(stacked[:, 2].min() - margin)), float(stacked[:, 2].max() + margin))
        set_axes_equal(ax)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Altitude Z")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    ax.view_init(elev=args.view_elev, azim=args.view_azim)
    ax.grid(True)

    info_text = (
        f"Task: {record['task_name']}\n"
        f"Episode: {record['episode_id']} | Steps: {record['episode_steps']}\n"
        f"Win: {record['win']} | Blue dead: {record['blue_dead']} | Red dead: {record['red_dead']}\n"
        f"Pincer score: {record['pincer_score']:.2f}"
    )

    ax.text2D(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=args.dpi)
    plt.close(fig)

    print(f"已保存三维轨迹图: {save_path}")


def save_record_json(record, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)

    print(f"已保存回合数据: {save_path}")


def prepare_agent_and_norm(args):
    check_model_dir(args.model_dir)

    env = MPEEnv(args)
    try:
        agent = build_agent(args, env)
        state_norm = load_state_norm(args)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    return agent, state_norm


def run_typical_win(args, agent, state_norm):
    print("\n" + "=" * 100)
    print("开始筛选图5-6：典型获胜回合三维轨迹图")
    print("=" * 100)

    records = collect_episodes(
        args=args,
        agent=agent,
        state_norm=state_norm,
        task_id=args.typical_task,
        mode="typical",
        num_episodes=args.num_episodes,
    )

    selected = select_typical_win(records)

    title = "Fig.5-6 Typical Winning Episode 3D Trajectory"
    save_path = Path(args.output_dir) / "fig5_6_typical_win.png"
    json_path = Path(args.output_dir) / "fig5_6_typical_win_record.json"

    plot_episode_3d(selected, save_path, title, args)
    save_record_json(selected, json_path)


def run_pincer(args, agent, state_norm):
    print("\n" + "=" * 100)
    print("开始筛选图5-7：侧翼压迫任务下的协同包夹轨迹图")
    print("=" * 100)

    records = collect_episodes(
        args=args,
        agent=agent,
        state_norm=state_norm,
        task_id=args.pincer_task,
        mode="pincer",
        num_episodes=args.num_episodes,
    )

    selected = select_best_pincer(records)

    title = "Fig.5-7 Cooperative Pincer Trajectory in Flank-Press Task"
    save_path = Path(args.output_dir) / "fig5_7_pincer.png"
    json_path = Path(args.output_dir) / "fig5_7_pincer_record.json"

    plot_episode_3d(selected, save_path, title, args)
    save_record_json(selected, json_path)


def run_failure(args, agent, state_norm):
    print("\n" + "=" * 100)
    print("开始筛选图5-8：单机失效场景下的剩余无人机追击轨迹图")
    print("=" * 100)

    records = collect_episodes(
        args=args,
        agent=agent,
        state_norm=state_norm,
        task_id=args.failure_task,
        mode="failure",
        num_episodes=args.num_episodes,
    )

    selected = select_best_failure_recovery(records)

    title = "Fig.5-8 Remaining UAV Pursuit Trajectory under Single-Agent Failure"
    save_path = Path(args.output_dir) / "fig5_8_failure_recovery.png"
    json_path = Path(args.output_dir) / "fig5_8_failure_recovery_record.json"

    plot_episode_3d(selected, save_path, title, args)
    save_record_json(selected, json_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="Meta-MAPPO")
    parser.add_argument("--model_dir", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--date", type=str, default="plot3d")
    parser.add_argument("--output_dir", type=str, default="./result/selected_3d")

    parser.add_argument("--figs", type=str, default="all",
                        help="all / typical / pincer / failure，可用逗号组合，如 typical,pincer")

    parser.add_argument("--typical_task", type=int, default=6)
    parser.add_argument("--pincer_task", type=int, default=6)
    parser.add_argument("--failure_task", type=int, default=9)

    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=256)

    parser.add_argument("--failure_step", type=int, default=50)
    parser.add_argument("--force_red0_failure", type=str2bool, default=True)

    parser.add_argument("--pincer_distance", type=float, default=5.0)
    parser.add_argument("--pincer_angle", type=float, default=60.0)

    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--deterministic", type=str2bool, default=True)

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

    parser.add_argument("--meta_support_envs", type=int, default=4)
    parser.add_argument("--meta_buffer_size", type=int, default=3200)
    parser.add_argument("--meta_inner_epochs", type=int, default=1)
    parser.add_argument("--meta_outer_epochs", type=int, default=1)

    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--view_elev", type=float, default=25.0)
    parser.add_argument("--view_azim", type=float, default=-60.0)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for task_id in [args.typical_task, args.pincer_task, args.failure_task]:
        if task_id not in TASKS:
            raise ValueError(f"不支持的 task_id={task_id}，当前支持: {sorted(TASKS.keys())}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("三维轨迹典型回合筛选与绘图")
    print(f"算法: {args.algo_name}")
    print(f"模型: {args.model_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 100)

    agent, state_norm = prepare_agent_and_norm(args)

    figs = [x.strip().lower() for x in args.figs.split(",") if x.strip()]
    if "all" in figs:
        figs = ["typical", "pincer", "failure"]

    if "typical" in figs:
        run_typical_win(args, agent, state_norm)

    if "pincer" in figs:
        run_pincer(args, agent, state_norm)

    if "failure" in figs:
        run_failure(args, agent, state_norm)

    print("\n全部三维轨迹图生成完成。")


if __name__ == "__main__":
    main()