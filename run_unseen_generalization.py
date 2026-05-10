import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


TASKS = {
    6: "U0_mixed_flank_then_decoy",
    7: "U1_stronger_dynamics",
    8: "U2_large_initial_perturbation",
    9: "U3_obs_noise_red_failure",
    10: "U4_weapon_parameter_shift",
}

ALGOS = ["MAPPO", "Meta-MAPPO"]


def run_cmd(cmd, cwd, dry_run=False):
    print("\n" + "=" * 100)
    print("运行命令：")
    print(" ".join(cmd))
    print("=" * 100)

    if dry_run:
        return

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"命令运行失败，返回码：{result.returncode}")


def check_model_dir(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")

    actor_path = model_dir / "actor_shared.pt"
    critic_path = model_dir / "critic_shared.pt"

    if not actor_path.exists():
        raise FileNotFoundError(f"缺少 actor_shared.pt：{actor_path}")

    if not critic_path.exists():
        raise FileNotFoundError(f"缺少 critic_shared.pt：{critic_path}")


def move_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


def evaluate_one_task(args, root: Path, algo_name: str, model_dir: Path, task_id: int):
    task_name = TASKS[task_id]

    cmd = [
        sys.executable,
        "evaluate.py",
        "--algo_name", algo_name,
        "--model_dir", str(model_dir),
        "--max_episode_steps", str(args.max_episode_steps),
        "--policy_dist", args.policy_dist,
        "--eval_task", str(task_id),
    ]

    run_cmd(cmd, cwd=root, dry_run=args.dry_run)

    if args.dry_run:
        return None

    output_dir = root / args.output_dir / "unseen_generalization" / algo_name / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        f"combat_metrics_{algo_name}.json",
        f"robustness_winrate_{algo_name}.npy",
        f"robustness_reward_{algo_name}.npy",
        f"robustness_data_{algo_name}.npy",
        f"recovery_data_{algo_name}.npy",
    ]

    for filename in generated_files:
        src = root / filename
        dst = output_dir / filename
        move_if_exists(src, dst)

    metrics_path = output_dir / f"combat_metrics_{algo_name}.json"

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        metrics["task_id"] = task_id
        metrics["task_name"] = task_name
        metrics["model_dir"] = str(model_dir)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        return metrics

    return None


def save_summary(root: Path, args, all_metrics):
    summary_dir = root / args.output_dir / "unseen_generalization"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / "unseen_generalization_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    csv_path = summary_dir / "unseen_generalization_summary.csv"

    headers = [
        "algo_name",
        "task_id",
        "task_name",
        "win_rate",
        "avg_kills",
        "survival_rate",
        "exchange_ratio",
        "avg_win_steps",
        "avg_energy",
        "total_kills",
        "total_deaths",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")

        for item in all_metrics:
            row = []
            for h in headers:
                row.append(str(item.get(h, "")))
            f.write(",".join(row) + "\n")

    print("\n汇总结果已保存：")
    print(summary_path)
    print(csv_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_episode_steps", type=int, default=256)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--output_dir", type=str, default="./result")

    parser.add_argument("--tasks", type=str, default="6,7,8,9,10")
    parser.add_argument("--algos", type=str, default="MAPPO,Meta-MAPPO")

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--mappo_model_dir", type=str, default="")
    parser.add_argument("--meta_mappo_model_dir", type=str, default="")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    task_ids = [int(x.strip()) for x in args.tasks.split(",") if x.strip()]
    algos = [x.strip() for x in args.algos.split(",") if x.strip()]

    for task_id in task_ids:
        if task_id not in TASKS:
            raise ValueError(f"不支持的未见任务 task_id={task_id}，应为 6~10。")

    for algo in algos:
        if algo not in ALGOS:
            raise ValueError(f"不支持的算法 {algo}，应为 MAPPO 或 Meta-MAPPO。")

    if not args.skip_train:
        raise ValueError(
            "这个脚本用于已经训练好模型后的 U0~U4 未见任务直接评估，"
            "请添加 --skip_train，并提供 --mappo_model_dir 和 --meta_mappo_model_dir。"
        )

    model_dirs = {}

    if "MAPPO" in algos:
        if not args.mappo_model_dir:
            raise ValueError("需要提供 --mappo_model_dir")
        model_dirs["MAPPO"] = Path(args.mappo_model_dir)

    if "Meta-MAPPO" in algos:
        if not args.meta_mappo_model_dir:
            raise ValueError("需要提供 --meta_mappo_model_dir")
        model_dirs["Meta-MAPPO"] = Path(args.meta_mappo_model_dir)

    if not args.dry_run:
        for algo_name, model_dir in model_dirs.items():
            check_model_dir(model_dir)

    all_metrics = []

    for algo_name in algos:
        model_dir = model_dirs[algo_name]

        for task_id in task_ids:
            metrics = evaluate_one_task(
                args=args,
                root=root,
                algo_name=algo_name,
                model_dir=model_dir,
                task_id=task_id,
            )

            if metrics is not None:
                all_metrics.append(metrics)

    if not args.dry_run:
        save_summary(root, args, all_metrics)

    print("\nU0~U4 未见任务泛化评估全部完成。")


if __name__ == "__main__":
    main()