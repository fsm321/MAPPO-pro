import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import copy
import torch
import numpy as np
import pyglet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env.MPE_env import MPEEnv
from utils.normalization import Normalization
from utils.replaybuffer import ReplayBuffer
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
import argparse
import json
from env.scenarios.air_combat_2v2 import (
    ALL_TASK_IDS,
    META_TRAIN_TASK_IDS,
    META_TEST_TASK_IDS,
)


def normalize_obs(args, obs, state_norm):
    if args.use_state_norm and state_norm is not None:
        return state_norm(obs, update=False)
    return obs


def build_eval_task_onehot(task_id, task_dim):
    """
    评估/快速适应阶段的 task code。

    训练任务 T0~T5：使用正常 one-hot。
    测试任务 U0~U4：使用全零向量，表示 unknown task。

    这样不会泄露 meta-test 任务标签，也不会触发
    get_train_task_encoder_index(task_id) 对 U0~U4 报错。
    """
    task_id = int(task_id)
    task_onehot = np.zeros(task_dim, dtype=np.float32)

    if task_id in META_TRAIN_TASK_IDS:
        task_onehot[META_TRAIN_TASK_IDS.index(task_id)] = 1.0

    return task_onehot


def build_eval_agent_onehot(agent_idx, n_red):
    agent_onehot = np.zeros(n_red, dtype=np.float32)
    agent_onehot[agent_idx] = 1.0
    return agent_onehot


def build_eval_shared_obs(red_obs, task_id, task_dim, agent_idx, n_red):
    """
    centralized critic 输入：
    红方联合观测 + task code + agent code
    """
    red_obs = np.asarray(red_obs, dtype=np.float32).reshape(-1)
    task_code = build_eval_task_onehot(task_id, task_dim)
    agent_code = build_eval_agent_onehot(agent_idx, n_red)
    return np.concatenate((red_obs, task_code, agent_code), axis=0)


def collect_support_buffer(args, env, agent, state_norm, task_id):
    """
    在指定 meta-test task 上采少量 support episodes，
    用于快速适应。

    注意：
    1. support 阶段使用 stochastic action，因为 PPO update 需要 old log_prob；
    2. query/evaluate 阶段使用 deterministic action；
    3. 不更新 state_norm，只使用训练时保存的归一化统计量；
    4. 这里只采红方两个智能体的数据，蓝方仍由规则策略控制。
    """
    red_ids = [0, 1]
    buffer_size = args.adapt_support_episodes * args.max_episode_steps * len(red_ids)

    buffer_args = copy.copy(args)
    buffer_args.buffer_size = max(buffer_size, len(red_ids))

    support_buffer = ReplayBuffer(buffer_args)

    for _ in range(args.adapt_support_episodes):
        s = env.reset(task_id=task_id)
        dones = np.zeros(env.n, dtype=bool)
        episode_steps = 0

        # 保留 inactive transition，确保 buffer 顺序仍然是
        # [red0, red1], [red0, red1]...，便于 rollout_group_size=2 的 GAE 分组。
        red_active_mask = np.ones(len(red_ids), dtype=bool)

        while (not np.all(dones)) and episode_steps < args.max_episode_steps:
            if support_buffer.count + len(red_ids) > buffer_args.buffer_size:
                break

            episode_steps += 1

            red_obs_norm = np.zeros((len(red_ids), args.state_dim), dtype=np.float32)
            for j, rid in enumerate(red_ids):
                red_obs_norm[j] = normalize_obs(args, s[rid], state_norm)

            actions = [np.zeros(args.action_dim, dtype=np.float32) for _ in range(env.n)]
            a_list = []
            logp_list = []

            # support 采样必须用随机动作，不能用 deterministic。
            for j, rid in enumerate(red_ids):
                if red_active_mask[j]:
                    a, a_logp = agent.choose_action(red_obs_norm[j])
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                else:
                    a = np.zeros(args.action_dim, dtype=np.float32)
                    a_logp = np.zeros(1, dtype=np.float32)
                    action = np.zeros(args.action_dim, dtype=np.float32)

                actions[rid] = action
                a_list.append(a)
                logp_list.append(a_logp)

            s_next, r, done, _ = env.step(actions)
            done = np.asarray(done, dtype=bool)

            red_next_obs_norm = np.zeros((len(red_ids), args.state_dim), dtype=np.float32)
            for j, rid in enumerate(red_ids):
                red_next_obs_norm[j] = normalize_obs(args, s_next[rid], state_norm)

            for j, rid in enumerate(red_ids):
                share_obs = build_eval_shared_obs(
                    red_obs_norm,
                    task_id,
                    args.task_dim,
                    agent_idx=j,
                    n_red=args.n_red
                )

                share_obs_next = build_eval_shared_obs(
                    red_next_obs_norm,
                    task_id,
                    args.task_dim,
                    agent_idx=j,
                    n_red=args.n_red
                )

                if red_active_mask[j]:
                    scaled_reward = float(r[rid]) * args.adapt_reward_scale
                    dw = bool(done[rid] and episode_steps != args.max_episode_steps)
                    done_for_gae = bool(done[rid] or episode_steps >= args.max_episode_steps)
                    active_mask = 1.0
                else:
                    scaled_reward = 0.0
                    dw = True
                    done_for_gae = True
                    active_mask = 0.0

                support_buffer.store(
                    red_obs_norm[j],
                    share_obs,
                    a_list[j],
                    logp_list[j],
                    scaled_reward,
                    red_next_obs_norm[j],
                    share_obs_next,
                    dw,
                    done_for_gae,
                    active_mask
                )

                if done[rid] and not np.all(done) and episode_steps < args.max_episode_steps:
                    red_active_mask[j] = False

            s = s_next
            dones = done

    return support_buffer


def adapt_agent_on_task(args, base_agent, state_norm, task_id):
    """
    复制 base_agent，在指定任务上用 support buffer 做少量 PPO inner update。
    原始 base_agent 不会被修改。
    """
    adapted_agent = copy.deepcopy(base_agent)
    support_env = MPEEnv(args)

    try:
        support_buffer = collect_support_buffer(
            args=args,
            env=support_env,
            agent=adapted_agent,
            state_norm=state_norm,
            task_id=task_id
        )
    finally:
        close_fn = getattr(support_env, "close", None)
        if callable(close_fn):
            close_fn()

    if support_buffer.count <= 0:
        print(f"[警告] task_id={task_id} 没有采到 support 数据，跳过适应。")
        return adapted_agent, 0

    adapted_agent.update(
        support_buffer,
        total_steps=0,
        do_lr_decay=False,
        rollout_group_size=args.n_red,
        K_epochs_override=args.adapt_epochs
    )

    return adapted_agent, support_buffer.count


def parse_task_ids(task_string):
    if task_string.strip().lower() == "all":
        return META_TEST_TASK_IDS
    return [int(x.strip()) for x in task_string.split(",") if x.strip()]


def evaluate_fast_adaptation(args, base_agent, state_norm):
    """
    对 U0~U4 做快速适应评估：
    1. 原模型直接评估；
    2. 复制模型；
    3. support episodes 上快速适应；
    4. query episodes 上评估适应后性能；
    5. 保存结果。
    """
    task_ids = parse_task_ids(args.adapt_tasks)
    results = []

    for task_id in task_ids:
        if task_id not in ALL_TASK_IDS:
            raise ValueError(f"Unknown task_id={task_id}, should be one of {ALL_TASK_IDS}")

        print("\n" + "=" * 70)
        print(f"开始快速适应评估: task_id={task_id}")
        print("=" * 70)

        direct_env = MPEEnv(args)
        direct_agents = [base_agent, base_agent, None, None]

        try:
            direct_metrics = evaluate_combat_metrics(
                args=args,
                env=direct_env,
                agents=direct_agents,
                state_norm=state_norm,
                times=args.adapt_query_episodes,
                task_id=task_id
            )
        finally:
            close_fn = getattr(direct_env, "close", None)
            if callable(close_fn):
                close_fn()

        (
            direct_win_rate,
            direct_exchange_ratio,
            direct_avg_win_steps,
            direct_avg_energy,
            direct_total_kills,
            direct_total_deaths
        ) = direct_metrics

        print(
            f"[未适应] task_id={task_id} | "
            f"胜率={direct_win_rate * 100:.1f}% | "
            f"战损比={direct_exchange_ratio:.2f}"
        )

        adapted_agent, support_count = adapt_agent_on_task(
            args=args,
            base_agent=base_agent,
            state_norm=state_norm,
            task_id=task_id
        )

        adapted_env = MPEEnv(args)
        adapted_agents = [adapted_agent, adapted_agent, None, None]

        try:
            adapted_metrics = evaluate_combat_metrics(
                args=args,
                env=adapted_env,
                agents=adapted_agents,
                state_norm=state_norm,
                times=args.adapt_query_episodes,
                task_id=task_id
            )
        finally:
            close_fn = getattr(adapted_env, "close", None)
            if callable(close_fn):
                close_fn()

        (
            adapted_win_rate,
            adapted_exchange_ratio,
            adapted_avg_win_steps,
            adapted_avg_energy,
            adapted_total_kills,
            adapted_total_deaths
        ) = adapted_metrics

        print(
            f"[适应后] task_id={task_id} | "
            f"support样本数={support_count} | "
            f"胜率={adapted_win_rate * 100:.1f}% | "
            f"战损比={adapted_exchange_ratio:.2f}"
        )

        results.append({
            "task_id": int(task_id),
            "support_episodes": int(args.adapt_support_episodes),
            "adapt_epochs": int(args.adapt_epochs),
            "support_samples": int(support_count),
            "direct_win_rate": direct_win_rate * 100.0,
            "direct_exchange_ratio": direct_exchange_ratio,
            "direct_avg_win_steps": direct_avg_win_steps,
            "direct_avg_energy": direct_avg_energy,
            "direct_total_kills": int(direct_total_kills),
            "direct_total_deaths": int(direct_total_deaths),
            "adapted_win_rate": adapted_win_rate * 100.0,
            "adapted_exchange_ratio": adapted_exchange_ratio,
            "adapted_avg_win_steps": adapted_avg_win_steps,
            "adapted_avg_energy": adapted_avg_energy,
            "adapted_total_kills": int(adapted_total_kills),
            "adapted_total_deaths": int(adapted_total_deaths),
            "win_rate_improvement": adapted_win_rate * 100.0 - direct_win_rate * 100.0,
            "exchange_ratio_improvement": adapted_exchange_ratio - direct_exchange_ratio,
        })

    save_name = f"fast_adaptation_metrics_{args.algo_name}.json"
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n快速适应评估结果已保存: {save_name}")
    return results


def evaluate_policy(args, env, agents, state_norm, seed=0, times=100, task_id=None):
    evaluate_rewards = []
    for _ in range(times):
        s = env.reset(task_id=task_id) if task_id is not None else env.reset()
        episode_steps = 0
        episode_reward = 0.0
        dones = np.zeros(env.n)

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = normalize_obs(args, s[agent_id], state_norm)
                    a = agents[agent_id].choose_action_deterministic(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)

    return np.mean(evaluate_rewards)


def evaluate_combat_metrics(args, env, agents, state_norm, times=100, task_id=None):
    total_wins = 0
    total_red_combat_deaths = 0
    total_blue_deaths = 0
    win_steps = []
    total_energy_consumed = []

    for _ in range(times):
        s = env.reset(task_id=task_id) if task_id is not None else env.reset()
        episode_steps = 0
        dones = np.zeros(env.n)
        episode_energy = 0.0
        episode_reward = 0.0

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = normalize_obs(args, s[agent_id], state_norm)
                    a = agents[agent_id].choose_action_deterministic(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)
                    episode_energy += np.linalg.norm(action)

            s_next, r, done, info = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        world = getattr(env, 'world', None) or getattr(env.env, 'world', None) or getattr(env.unwrapped, 'world', None)

        if world is not None:
            red_dead = sum([
                1 for a in world.agents
                if a.team == 0 and getattr(a, 'is_dead', False) and getattr(a, 'hp', 100) <= 0
            ])
            blue_dead = sum([
                1 for a in world.agents
                if a.team == 1 and getattr(a, 'is_dead', False)
            ])
        else:
            red_dead = sum([int(d) for d in dones[:2]])
            blue_dead = 2 if episode_reward > 80 else (1 if episode_reward > 30 else 0)

        total_red_combat_deaths += red_dead
        total_blue_deaths += blue_dead
        total_energy_consumed.append(episode_energy)

        is_win = (blue_dead == 2) or (blue_dead > red_dead)
        if is_win:
            total_wins += 1
            win_steps.append(episode_steps)

    win_rate = total_wins / times
    exchange_ratio = total_blue_deaths / max(total_red_combat_deaths, 1e-5)
    avg_win_steps = np.mean(win_steps) if len(win_steps) > 0 else args.max_episode_steps
    avg_energy = np.mean(total_energy_consumed)

    return win_rate, exchange_ratio, avg_win_steps, avg_energy, total_blue_deaths, total_red_combat_deaths


def evaluate_robustness(args, env, agents, state_norm, noise_std, times=100, task_id=None):
    total_wins = 0
    evaluate_rewards = []

    for _ in range(times):
        s = env.reset(task_id=task_id) if task_id is not None else env.reset()
        episode_steps = 0
        episode_reward = 0.0
        dones = np.zeros(env.n)

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            noisy_s = [
                state + np.random.normal(0, noise_std, size=state.shape)
                for state in s
            ]

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = normalize_obs(args, noisy_s[agent_id], state_norm)
                    a = agents[agent_id].choose_action_deterministic(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)

        world = getattr(env, 'world', None) or getattr(env.env, 'world', None) or getattr(env.unwrapped, 'world', None)

        if world is not None:
            red_dead = sum([
                1 for a in world.agents
                if a.team == 0 and getattr(a, 'is_dead', False) and getattr(a, 'hp', 100) <= 0
            ])
            blue_dead = sum([
                1 for a in world.agents
                if a.team == 1 and getattr(a, 'is_dead', False)
            ])
        else:
            red_dead = sum([int(d) for d in dones[:2]])
            blue_dead = 2 if episode_reward > 80 else (1 if episode_reward > 30 else 0)

        is_win = (blue_dead == 2) or (blue_dead > red_dead)
        if is_win:
            total_wins += 1

    win_rate = total_wins / times * 100.0
    avg_reward = np.mean(evaluate_rewards)
    return win_rate, avg_reward


def evaluate_failure_recovery(args, env, agents, state_norm, task_id=None):
    s = env.reset(task_id=task_id) if task_id is not None else env.reset()
    step_rewards = []
    dones = np.zeros(env.n)

    for step in range(args.max_episode_steps):
        actions = []

        for agent_id in range(env.n):
            if agents[agent_id] is None:
                actions.append(np.zeros(args.action_dim))
            elif agent_id == 0 and step >= 50:
                actions.append(np.zeros(args.action_dim))
                s[agent_id] = np.zeros_like(s[agent_id])
            else:
                obs = normalize_obs(args, s[agent_id], state_norm)
                a = agents[agent_id].choose_action_deterministic(obs)
                action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                actions.append(action)

        s_next, r, done, _ = env.step(actions)
        step_rewards.append(sum(r[:2]))
        s = s_next

        if all(done):
            step_rewards.extend([0] * (args.max_episode_steps - len(step_rewards)))
            break

    return step_rewards


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True

    if v.lower() in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")

    parser.add_argument("--date", type=str, default="eval")
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--model_dir", type=str, default="./data/model_to_eval")
    parser.add_argument("--policy_dist", type=str, default="Gaussian")

    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=5e8)

    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--mini_batch_size", type=int, default=1000)

    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--lr_a", type=float, default=3e-5)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.03)

    parser.add_argument("--use_adv_norm", type=str2bool, default=True)
    parser.add_argument("--use_state_norm", type=str2bool, default=True)
    parser.add_argument("--use_lr_decay", type=str2bool, default=True)
    parser.add_argument("--use_grad_clip", type=str2bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True)
    parser.add_argument("--set_adam_eps", type=str2bool, default=True)
    parser.add_argument("--use_tanh", type=str2bool, default=True)
    parser.add_argument("--eval_task", type=int, default=-1)
    parser.add_argument("--eval_adaptation", action="store_true")
    parser.add_argument("--only_adaptation", action="store_true")
    parser.add_argument("--adapt_tasks", type=str, default="6,7,8,9,10")
    parser.add_argument("--adapt_support_episodes", type=int, default=5)
    parser.add_argument("--adapt_query_episodes", type=int, default=100)
    parser.add_argument("--adapt_epochs", type=int, default=3)
    parser.add_argument("--adapt_reward_scale", type=float, default=0.1)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = MPEEnv(args)
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])
    args.n_red = getattr(args, "n_red", 2)
    args.task_dim = len(META_TRAIN_TASK_IDS)
    args.share_state_dim = args.state_dim * args.n_red + args.task_dim + args.n_red

    if args.algo_name == "Meta-MAPPO":
        shared_agent = Meta_MAPPO_Continuous(args)
    else:
        shared_agent = MAPPO_Continuous(args)

    print(f"正在加载 {args.algo_name} 模型...")
    try:
        shared_agent.restore(0)
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败，请检查路径。报错信息: {e}")
        raise SystemExit(1)

    agents = [shared_agent, shared_agent, None, None]

    state_norm = None
    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)
        try:
            state_norm.running_ms.mean = np.load(f"{args.model_dir}/norm_mean.npy")
            state_norm.running_ms.std = np.load(f"{args.model_dir}/norm_std.npy")
            print("状态归一化统计量加载成功。")
        except FileNotFoundError:
            print("未找到 norm_mean.npy / norm_std.npy，将按原始状态评估。")
            state_norm = None

    if args.only_adaptation:
        print("\n--- 只运行 Meta-test 快速适应评估 ---")
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        evaluate_fast_adaptation(args, shared_agent, state_norm)
        raise SystemExit(0)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_winrates = []
    robustness_rewards = []
    eval_task = None if args.eval_task < 0 else int(args.eval_task)

    if eval_task is not None and eval_task not in ALL_TASK_IDS:
        raise ValueError(
            f"--eval_task must be one of {ALL_TASK_IDS}, got {eval_task}."
        )

    if eval_task in META_TEST_TASK_IDS:
        print(f"当前评估任务: U{eval_task - META_TEST_TASK_IDS[0]} (task_id={eval_task})")
    elif eval_task in META_TRAIN_TASK_IDS:
        print(f"当前评估任务: T{eval_task} (task_id={eval_task})")

    print("\n--- 开始进行鲁棒性评估 ---")
    for noise in noise_levels:
        win_rate, avg_reward = evaluate_robustness(
            args,
            env,
            agents,
            state_norm,
            noise_std=noise,
            times=100,
            task_id=eval_task
        )

        robustness_winrates.append(win_rate)
        robustness_rewards.append(avg_reward)

        print(
            f"噪声强度 {noise:.1f} -> "
            f"胜率: {win_rate:.1f}% | 平均奖励: {avg_reward:.2f}"
        )

    np.save(f"robustness_winrate_{args.algo_name}.npy", robustness_winrates)
    np.save(f"robustness_reward_{args.algo_name}.npy", robustness_rewards)
    np.save(f"robustness_data_{args.algo_name}.npy", robustness_rewards)

    print("\n--- 开始进行失效恢复测试 ---")
    recovery_curve = evaluate_failure_recovery(args, env, agents, state_norm, task_id=eval_task)
    np.save(f"recovery_data_{args.algo_name}.npy", recovery_curve)

    print("\n--- 开始进行 100 局高阶空战效能评估 ---")
    win_rate, exchange_ratio, avg_win_steps, avg_energy, total_kills, total_deaths = evaluate_combat_metrics(
        args, env, agents, state_norm, times=100, task_id=eval_task
    )

    print("\n=======================================================")
    print(f"        >>> {args.algo_name} 最终战术效能评估报告 <<<")
    print("=======================================================")
    print(f"1. 综合任务胜率 (Win Rate):           {win_rate * 100:.1f} %")
    print(
        f"2. 战损交换比 (Loss-Exchange Ratio):  {exchange_ratio:.2f} "
        f"(共击落 {total_kills} 架 / 阵亡 {total_deaths} 架)"
    )
    print(f"3. 获胜平均耗时 (Avg Time-to-Kill):   {avg_win_steps:.1f} 步")
    print(f"4. 机动能量消耗 (Maneuver Energy):    {avg_energy:.1f}")
    print("=======================================================\n")

    num_eval_episodes = 100
    num_red = 2

    combat_metrics = {
        "algo_name": args.algo_name,
        "num_eval_episodes": num_eval_episodes,
        "avg_kills": total_kills / num_eval_episodes,
        "survival_rate": (num_red * num_eval_episodes - total_deaths) / (num_red * num_eval_episodes) * 100.0,
        "exchange_ratio": exchange_ratio,
        "avg_win_steps": avg_win_steps,
        "avg_energy": avg_energy,
        "total_kills": int(total_kills),
        "total_deaths": int(total_deaths),
        "win_rate": win_rate * 100.0
    }

    with open(f"combat_metrics_{args.algo_name}.json", "w", encoding="utf-8") as f:
        json.dump(combat_metrics, f, ensure_ascii=False, indent=4)

    print(f"战术效能指标已保存: combat_metrics_{args.algo_name}.json")

    if args.eval_adaptation:
        print("\n--- 开始 Meta-test 快速适应评估 ---")
        evaluate_fast_adaptation(args, shared_agent, state_norm)
