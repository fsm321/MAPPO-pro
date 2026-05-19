import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth_data(scalars, weight):
    if weight <= 0.0:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def get_data_from_base_dir(base_dir, keyword):
    if not os.path.exists(base_dir):
        print(f"路径不存在，跳过: {base_dir}")
        return None, None

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                try:
                    ea = EventAccumulator(root)
                    ea.Reload()
                    tags = ea.Tags().get('scalars', [])
                    for tag in tags:
                        if keyword.lower() in tag.lower():
                            events = ea.Scalars(tag)
                            steps = [e.step for e in events]
                            values = [e.value for e in events]
                            return np.array(steps), np.array(values)
                except Exception as e:
                    continue
    return None, None

def plot_task_winrate(algo_logs, task_id, smooth_weight=0.98):
    plt.figure(figsize=(10, 6))
    colors = {
        "MetaMAPPO": "#DD8452",
        "MAPPO": "#4C72B0"
    }
    keyword = f"Task_{task_id}_WinRate"
    has_data = False

    for algo, dirs in algo_logs.items():
        all_steps = []
        all_values_raw = []
        all_values_smoothed = []

        for d in dirs:
            steps, vals = get_data_from_base_dir(d, keyword)
            if vals is not None:
                vals_smoothed = smooth_data(vals, weight=smooth_weight)
                all_steps.append(steps)
                all_values_raw.append(vals)
                all_values_smoothed.append(vals_smoothed)

        if all_values_raw:
            has_data = True
            min_len = min([len(v) for v in all_values_raw])
            steps_clipped = all_steps[0][:min_len]
            raw_clipped = [v[:min_len] for v in all_values_raw]
            smoothed_clipped = [v[:min_len] for v in all_values_smoothed]

            mean_raw = np.mean(raw_clipped, axis=0)
            mean_smoothed = np.mean(smoothed_clipped, axis=0)
            std_smoothed = np.std(smoothed_clipped, axis=0)

            color = colors.get(algo, "#000000")
            plt.fill_between(steps_clipped,
                             mean_smoothed - std_smoothed,
                             mean_smoothed + std_smoothed,
                             alpha=0.2, color=color)
            if smooth_weight > 0:
                plt.plot(steps_clipped, mean_raw, alpha=0.2, color=color, linewidth=1)
            plt.plot(steps_clipped, mean_smoothed, label=algo, linewidth=1.5, color=color)

    if has_data:
        plt.xlabel('Step', fontsize=14, fontweight='bold')
        plt.ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
        plt.title(f'Task_{task_id} Win Rate', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        output_name = f"Task_{task_id}_WinRate.png"
        plt.savefig(output_name, dpi=300)
        print(f"🎉 成功生成: {output_name}")
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    algo_logs = {
        "MetaMAPPO": [r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\MetaMAPPO_stable_seed10_lr0005"],
        "MAPPO": [r"D:\Meta-MAPPO\Meta-MAPPO\10.0\data\Meta_MAPPO_greedy_inner2"]
    }

    for task_id in range(6):
        plot_task_winrate(algo_logs, task_id)