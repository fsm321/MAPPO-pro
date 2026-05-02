import math
import numpy as np

from env._mpe_utils.core import Agent, World, fast_compute_distance_and_angle_scalar
from env._mpe_utils.scenario import BaseScenario


META_TRAIN_TASK_IDS = [0, 1, 2, 3, 4, 5]
META_TEST_TASK_IDS = [6, 7, 8, 9, 10]
ALL_TASK_IDS = META_TRAIN_TASK_IDS + META_TEST_TASK_IDS
TRAIN_TASK_ENCODER_INDEX = {
    task_id: idx for idx, task_id in enumerate(META_TRAIN_TASK_IDS)
}


def get_train_task_encoder_index(task_id):
    """
    Centralized critic uses a 6-dim task code for meta-training tasks only.
    Keep the environment task id separate so meta-test tasks never get silently clipped.
    """
    task_id = int(task_id)
    if task_id not in TRAIN_TASK_ENCODER_INDEX:
        raise ValueError(
            "Centralized critic task encoder only supports meta-training task ids "
            f"{META_TRAIN_TASK_IDS}, got {task_id}."
        )
    return TRAIN_TASK_ENCODER_INDEX[task_id]

TASK_CONFIGS = {
    0: {
        "name": "T0_head_on_chase",
        "blue_tactics": [0, 0],
        "blue_noise": 0.10,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "head_on",
    },
    1: {
        "name": "T1_flank_and_press",
        "blue_tactics": [2, 0],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "flank",
    },
    2: {
        "name": "T2_high_low_layer",
        "blue_tactics": [3, 4],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.6,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "altitude_layer",
    },
    3: {
        "name": "T3_decoy_and_attack",
        "blue_tactics": [5, 0],
        "blue_noise": 0.10,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "decoy",
    },
    4: {
        "name": "T4_noisy_chase",
        "blue_tactics": [0, 0],
        "blue_noise": 0.30,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "head_on",
    },
    5: {
        "name": "T5_defensive_counter",
        "blue_tactics": [6, 6],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.7,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "defensive",
    },
    6: {
        "name": "U0_mixed_flank_then_decoy",
        "blue_tactics": [2, 0],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "flank",
        "switch_step": 250,
        "switch_tactics": [5, 0],
    },
    7: {
        "name": "U1_stronger_dynamics",
        "blue_tactics": [0, 2],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.15,
        "blue_turn_scale": 1.15,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "head_on",
    },
    8: {
        "name": "U2_large_initial_perturbation",
        "blue_tactics": [0, 2],
        "blue_noise": 0.15,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "large_perturbation",
    },
    9: {
        "name": "U3_obs_noise_red_failure",
        "blue_tactics": [0, 2],
        "blue_noise": 0.18,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 3.5,
        "blue_attack_angle": math.pi / 6,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "head_on",
        "obs_noise_std": 0.10,
        "red_failure_step": 50,
        "failed_red_id": 0,
    },
    10: {
        "name": "U4_weapon_parameter_shift",
        "blue_tactics": [0, 2],
        "blue_noise": 0.12,
        "blue_speed_scale": 1.0,
        "blue_turn_scale": 1.0,
        "blue_attack_range": 4.2,
        "blue_attack_angle": math.pi / 8,
        "red_attack_range": 4.5,
        "red_attack_angle": math.pi / 4,
        "init_mode": "head_on",
    },
}


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.num_agents, world.collaborative = 4, False
        self.current_task = META_TRAIN_TASK_IDS[0]
        self.task_cfg = TASK_CONFIGS[self.current_task]
        world.current_task = self.current_task
        world.task_cfg = self.task_cfg
        world.step_count = 0
        world.blue_tactics_assigned = False
        world.blue_tactics_signature = None
        world.red_failure_applied = False
        world.agents = [Agent() for _ in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name, agent.team = f"uav_{i}", (0 if i < 2 else 1)
            if agent.team == 1:
                agent.action_callback = self.blue_action_callback
        return world

    def _reset_agent_state_by_task(self, agent, idx, cfg):
        init_mode = cfg.get("init_mode", "head_on")

        if init_mode == "head_on":
            if agent.team == 0:
                agent.state.p_pos = np.array([
                    np.random.uniform(-5.0, -3.0),
                    np.random.uniform(-2.0, 2.0),
                ])
                agent.state.yaw = np.random.uniform(-0.2, 0.2)
            else:
                agent.state.p_pos = np.array([
                    np.random.uniform(3.0, 5.0),
                    np.random.uniform(-2.0, 2.0),
                ])
                agent.state.yaw = math.pi + np.random.uniform(-0.2, 0.2)
            agent.state.z_pos = np.random.uniform(3.5, 6.5)

        elif init_mode == "flank":
            if agent.team == 0:
                agent.state.p_pos = np.array([
                    np.random.uniform(-5.0, -3.0),
                    np.random.uniform(-1.5, 1.5),
                ])
                agent.state.yaw = np.random.uniform(-0.3, 0.3)
            else:
                flank_y = 4.0 if idx % 2 == 0 else -4.0
                agent.state.p_pos = np.array([
                    np.random.uniform(2.0, 4.0),
                    flank_y + np.random.uniform(-0.5, 0.5),
                ])
                agent.state.yaw = math.pi + np.random.uniform(-0.4, 0.4)
            agent.state.z_pos = np.random.uniform(3.5, 6.5)

        elif init_mode == "altitude_layer":
            if agent.team == 0:
                agent.state.p_pos = np.array([
                    np.random.uniform(-5.0, -3.0),
                    np.random.uniform(-2.0, 2.0),
                ])
                agent.state.z_pos = np.random.uniform(4.0, 6.0)
                agent.state.yaw = np.random.uniform(-0.2, 0.2)
            else:
                agent.state.p_pos = np.array([
                    np.random.uniform(2.5, 5.0),
                    np.random.uniform(-2.0, 2.0),
                ])
                if idx % 2 == 0:
                    agent.state.z_pos = np.random.uniform(6.5, 8.0)
                else:
                    agent.state.z_pos = np.random.uniform(2.0, 3.5)
                agent.state.yaw = math.pi + np.random.uniform(-0.3, 0.3)

        elif init_mode == "large_perturbation":
            if agent.team == 0:
                agent.state.p_pos = np.array([
                    np.random.uniform(-7.0, -2.0),
                    np.random.uniform(-5.0, 5.0),
                ])
                agent.state.z_pos = np.random.uniform(2.0, 8.0)
                agent.state.yaw = np.random.uniform(-0.6, 0.6)
            else:
                agent.state.p_pos = np.array([
                    np.random.uniform(2.0, 7.0),
                    np.random.uniform(-5.0, 5.0),
                ])
                agent.state.z_pos = np.random.uniform(2.0, 8.0)
                agent.state.yaw = math.pi + np.random.uniform(-0.6, 0.6)

        elif init_mode == "defensive":
            if agent.team == 0:
                agent.state.p_pos = np.array([
                    np.random.uniform(-4.5, -2.5),
                    np.random.uniform(-2.0, 2.0),
                ])
                agent.state.yaw = np.random.uniform(-0.2, 0.2)
            else:
                agent.state.p_pos = np.array([
                    np.random.uniform(4.0, 6.0),
                    np.random.uniform(-3.0, 3.0),
                ])
                agent.state.yaw = math.pi + np.random.uniform(-0.2, 0.2)
            agent.state.z_pos = np.random.uniform(3.5, 6.5)

        else:
            agent.state.p_pos = np.random.uniform(-5.0, 5.0, 2)
            agent.state.z_pos = np.random.uniform(3.0, 7.0)
            agent.state.yaw = np.random.uniform(-math.pi, math.pi)

        agent.state.p_vel = 1.0
        agent.state.pitch = 0.0
        agent.state.roll = 0.0

    def reset_world(self, world, task_id=None):
        if task_id is None:
            self.current_task = int(np.random.choice(META_TRAIN_TASK_IDS))
        else:
            self.current_task = int(task_id)

        if self.current_task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {self.current_task}")

        self.task_cfg = TASK_CONFIGS[self.current_task]
        world.current_task = self.current_task
        world.task_cfg = self.task_cfg
        world.step_count = 0
        world.blue_tactics_assigned = False
        world.blue_tactics_signature = None
        world.red_failure_applied = False

        for i, agent in enumerate(world.agents):
            if hasattr(agent, "last_min_dist"):
                delattr(agent, "last_min_dist")
            agent.hp, agent.is_dead, agent.last_action, agent.done = 100.0, False, np.zeros(3), False
            agent.just_killed_by_enemy = False
            agent.hit_enemy_this_step = False
            agent.kill_enemy_this_step = False
            agent.killed_by_enemy_this_step = False
            agent.won_this_step = False
            if agent.team == 1:
                agent.combat_mode, agent.tactic = False, None
            agent.color = np.array([0.85, 0.35, 0.35]) if agent.team == 0 else np.array([0.35, 0.35, 0.85])
            self._reset_agent_state_by_task(agent, i, self.task_cfg)

    def _get_current_blue_tactics(self, world, cfg):
        tactics = cfg.get("blue_tactics", [0, 0])
        switch_step = cfg.get("switch_step", None)
        if switch_step is not None and getattr(world, "step_count", 0) >= switch_step:
            tactics = cfg.get("switch_tactics", tactics)
        return tactics

    def _assign_blue_tactics(self, world, cfg):
        tactics = self._get_current_blue_tactics(world, cfg)
        signature = tuple(tactics)
        if getattr(world, "blue_tactics_signature", None) == signature:
            return

        blues = [a for a in world.agents if a.team == 1 and not a.is_dead]
        for i, blue in enumerate(blues):
            blue.tactic = tactics[i] if i < len(tactics) else tactics[-1]

        world.blue_tactics_signature = signature
        world.blue_tactics_assigned = True

    def blue_action_callback(self, agent, world):
        action = np.zeros(3)
        red_agents = [a for a in world.agents if a.team == 0 and not a.is_dead]
        if agent.is_dead or not red_agents:
            return action

        cfg = getattr(world, "task_cfg", TASK_CONFIGS[META_TRAIN_TASK_IDS[0]])
        my_pos = (agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos)
        dists = [
            math.sqrt(
                (r.state.p_pos[0] - my_pos[0]) ** 2 +
                (r.state.p_pos[1] - my_pos[1]) ** 2 +
                (r.state.z_pos - my_pos[2]) ** 2
            )
            for r in red_agents
        ]
        min_dist, closest_red = min(dists), red_agents[np.argmin(dists)]

        if min_dist > 6.0 and not agent.combat_mode:
            action[:] = [1.0, 0.0, 0.0]
        else:
            agent.combat_mode = True
            self._assign_blue_tactics(world, cfg)

            rel_p = closest_red.state.p_pos - agent.state.p_pos
            t_yaw = math.atan2(rel_p[1], rel_p[0])
            y_diff = (t_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
            z_diff = closest_red.state.z_pos - agent.state.z_pos

            roll_cmd = np.clip(y_diff / (math.pi / 2), -1.0, 1.0)
            nz_cmd = 0.5 if z_diff > 0 else -0.5

            if agent.tactic == 0:
                action[:] = [0.8, nz_cmd, roll_cmd]

            elif agent.tactic == 2:
                f_yaw = t_yaw + math.pi / 2
                fy_diff = (f_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
                flank_roll = np.clip(fy_diff / (math.pi / 2), -1.0, 1.0)
                action[:] = [0.8, 0.0, flank_roll]

            elif agent.tactic == 3:
                altitude_cmd = 0.6 if agent.state.z_pos < 7.0 else 0.0
                action[:] = [0.6, altitude_cmd, roll_cmd]

            elif agent.tactic == 4:
                altitude_cmd = -0.6 if agent.state.z_pos > 2.5 else 0.0
                action[:] = [0.9, altitude_cmd, roll_cmd]

            elif agent.tactic == 5:
                decoy_yaw = t_yaw + math.pi / 3
                decoy_diff = (decoy_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
                decoy_roll = np.clip(decoy_diff / (math.pi / 2), -1.0, 1.0)

                if min_dist < 4.0:
                    action[:] = [-0.2, 0.0, -decoy_roll]
                else:
                    action[:] = [0.5, 0.0, decoy_roll]

            elif agent.tactic == 6:
                if min_dist < 5.0:
                    evade_yaw = t_yaw + math.pi
                    evade_diff = (evade_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
                    evade_roll = np.clip(evade_diff / (math.pi / 2), -1.0, 1.0)
                    action[:] = [0.4, nz_cmd, evade_roll]
                else:
                    action[:] = [0.7, nz_cmd, roll_cmd]

            else:
                action[:] = [0.8, nz_cmd, roll_cmd]

        action[0] *= cfg.get("blue_speed_scale", 1.0)
        action[1] *= cfg.get("blue_turn_scale", 1.0)
        action[2] *= cfg.get("blue_turn_scale", 1.0)

        noise_std = cfg.get("blue_noise", 0.15)
        action += np.random.normal(0, noise_std, size=3)
        return np.clip(action, -1.0, 1.0)

    def _mark_red_team_done(self, world):
        for teammate in world.agents:
            if teammate.team == 0 and not teammate.is_dead:
                teammate.done = True
                teammate.won_this_step = True

    def _reset_step_event_flags(self, world):
        for agent in world.agents:
            agent.hit_enemy_this_step = False
            agent.kill_enemy_this_step = False
            agent.killed_by_enemy_this_step = False
            agent.won_this_step = False
            agent.just_killed_by_enemy = False

    def _select_attack_target(self, attacker, world):
        if attacker.is_dead:
            return None

        targets = [a for a in world.agents if a.team != attacker.team and not a.is_dead]
        if not targets:
            return None

        cfg = getattr(world, "task_cfg", TASK_CONFIGS[META_TRAIN_TASK_IDS[0]])
        if attacker.team == 1:
            attack_range = cfg.get("blue_attack_range", 3.5)
            attack_angle = cfg.get("blue_attack_angle", math.pi / 6)
        else:
            attack_range = cfg.get("red_attack_range", 4.5)
            attack_angle = cfg.get("red_attack_angle", math.pi / 4)

        target = min(
            targets,
            key=lambda other: math.sqrt(
                (other.state.p_pos[0] - attacker.state.p_pos[0]) ** 2 +
                (other.state.p_pos[1] - attacker.state.p_pos[1]) ** 2 +
                (other.state.z_pos - attacker.state.z_pos) ** 2
            )
        )
        distance, ata = fast_compute_distance_and_angle_scalar(
            attacker.state.p_pos[0], attacker.state.p_pos[1], attacker.state.z_pos,
            target.state.p_pos[0], target.state.p_pos[1], target.state.z_pos,
            attacker.state.yaw, attacker.state.pitch
        )
        if distance < attack_range and ata < attack_angle:
            return target, distance, ata
        return None

    def resolve_combat(self, world):
        world.step_count = getattr(world, "step_count", 0) + 1
        cfg = getattr(world, "task_cfg", TASK_CONFIGS[META_TRAIN_TASK_IDS[0]])
        self._reset_step_event_flags(world)

        failure_step = cfg.get("red_failure_step", None)
        failed_red_id = cfg.get("failed_red_id", None)
        if (
            failure_step is not None
            and not getattr(world, "red_failure_applied", False)
            and world.step_count >= failure_step
        ):
            red_agents = [a for a in world.agents if a.team == 0]
            if failed_red_id is not None and 0 <= failed_red_id < len(red_agents):
                failed_agent = red_agents[failed_red_id]
                failed_agent.is_dead = True
                failed_agent.done = True
                failed_agent.hp = 0.0
            world.red_failure_applied = True

        pending_damage = {}
        attackers_by_target = {}
        live_attackers = [agent for agent in world.agents if not agent.is_dead]

        for attacker in live_attackers:
            attack_event = self._select_attack_target(attacker, world)
            if attack_event is None:
                continue

            target, distance, ata = attack_event
            attacker.hit_enemy_this_step = True
            pending_damage[target] = pending_damage.get(target, 0.0) + 20.0
            attackers_by_target.setdefault(target, []).append((attacker, ata, distance, attacker.name))

        for target, damage in pending_damage.items():
            if target.is_dead:
                continue

            target.hp -= damage
            if target.hp <= 0:
                target.is_dead = True
                target.done = True
                target.killed_by_enemy_this_step = True
                target.just_killed_by_enemy = True
                killer = min(attackers_by_target[target], key=lambda item: (item[1], item[2], item[3]))[0]
                killer.kill_enemy_this_step = True

        blue_alive = [agent for agent in world.agents if agent.team == 1 and not agent.is_dead]
        if not blue_alive:
            self._mark_red_team_done(world)

    def reward(self, agent, world):
        if agent.is_dead:
            if getattr(agent, "killed_by_enemy_this_step", False):
                return -20.0
            return 0.0

        if agent.team == 1:
            return 0.0

        rew = 0.0

        if abs(agent.state.p_pos[0]) > 8.0 or abs(agent.state.p_pos[1]) > 8.0 or agent.state.z_pos < 1.0 or agent.state.z_pos > 9.0:
            rew -= 0.1

        rew -= 0.03

        if hasattr(agent.action, "u"):
            rew -= 0.02 * np.sum(np.square(agent.action.u - agent.last_action))

        ens = [e for e in world.agents if e.team == 1 and not e.is_dead]
        if not ens:
            if getattr(agent, "hit_enemy_this_step", False):
                rew += 15.0
                if getattr(agent, "kill_enemy_this_step", False):
                    rew += 120.0
            rew += 60.0
            return np.clip(rew, -25.0, 80.0)

        d_min, t_en = min(
            [
                (
                    math.sqrt(
                        (e.state.p_pos[0] - agent.state.p_pos[0]) ** 2 +
                        (e.state.p_pos[1] - agent.state.p_pos[1]) ** 2 +
                        (e.state.z_pos - agent.state.z_pos) ** 2
                    ),
                    e,
                )
                for e in ens
            ],
            key=lambda x: x[0]
        )
        _, ata = fast_compute_distance_and_angle_scalar(
            agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos,
            t_en.state.p_pos[0], t_en.state.p_pos[1], t_en.state.z_pos,
            agent.state.yaw, agent.state.pitch
        )

        is_engaging = d_min < 10.0
        am_i_attacking = False

        if is_engaging and (agent.state.z_pos < 3.0 or agent.state.z_pos > 7.0):
            rew -= 0.2

        if d_min < 15.0:
            dist_factor = max(0.0, (15.0 - d_min) / 15.0)
            rew += (math.pi - ata) / math.pi * 2.0 * dist_factor
            if getattr(agent, "hit_enemy_this_step", False):
                rew += 15.0
                am_i_attacking = True
                if getattr(agent, "kill_enemy_this_step", False):
                    rew += 120.0

        teammates = [a for a in world.agents if a.team == agent.team and a != agent and not a.is_dead]
        for teammate in teammates:
            dist_to_tm = math.sqrt(
                (agent.state.p_pos[0] - teammate.state.p_pos[0]) ** 2 +
                (agent.state.p_pos[1] - teammate.state.p_pos[1]) ** 2 +
                (agent.state.z_pos - teammate.state.z_pos) ** 2
            )

            if not am_i_attacking:
                if dist_to_tm < 0.5:
                    rew -= 1.5
                elif 1.5 < dist_to_tm < 4.0 and is_engaging:
                    rew += 0.5
            else:
                if dist_to_tm < 0.3:
                    rew -= 2.0

            if am_i_attacking:
                _, tm_ata = fast_compute_distance_and_angle_scalar(
                    teammate.state.p_pos[0], teammate.state.p_pos[1], teammate.state.z_pos,
                    t_en.state.p_pos[0], t_en.state.p_pos[1], t_en.state.z_pos,
                    teammate.state.yaw, teammate.state.pitch
                )
                dist_to_en = math.sqrt(
                    (t_en.state.p_pos[0] - teammate.state.p_pos[0]) ** 2 +
                    (t_en.state.p_pos[1] - teammate.state.p_pos[1]) ** 2 +
                    (t_en.state.z_pos - teammate.state.z_pos) ** 2
                )
                if dist_to_en < 6.0 and tm_ata < math.pi / 4:
                    rew += 10.0

        return np.clip(rew, -25.0, 80.0)

    def observation(self, agent, world):
        self_obs = [
            agent.state.p_pos[0],
            agent.state.p_pos[1],
            agent.state.z_pos,
            agent.state.p_vel,
            agent.state.yaw,
            agent.state.pitch,
            agent.state.roll,
        ]
        other_obs = []
        for other in world.agents:
            if other is agent:
                continue
            rel = other.state.p_pos - agent.state.p_pos
            rel_z = other.state.z_pos - agent.state.z_pos
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel_z ** 2)
            other_obs.extend([rel[0], rel[1], rel_z, dist, (0.0 if other.is_dead else 1.0)])

        obs = np.concatenate((self_obs, other_obs))
        cfg = getattr(world, "task_cfg", {})
        noise_std = cfg.get("obs_noise_std", 0.0)
        if noise_std > 0.0 and agent.team == 0:
            obs = obs + np.random.normal(0.0, noise_std, size=obs.shape)
        return obs

    def done(self, agent, world):
        return True if agent.is_dead else agent.done
