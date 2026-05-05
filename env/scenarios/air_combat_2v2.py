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

    def _wrap_angle(self, angle):
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _scale_blue_action(self, action, cfg):
        """
        Scale a blue-team action with the existing task-level multipliers.
        This keeps the original blue_speed_scale / blue_turn_scale behavior.
        """
        action = np.asarray(action, dtype=np.float32).copy()

        action[0] *= cfg.get("blue_speed_scale", 1.0)
        action[1] *= cfg.get("blue_turn_scale", 1.0)
        action[2] *= cfg.get("blue_turn_scale", 1.0)

        return np.clip(action, -1.0, 1.0)

    def _predict_blue_next_state(self, agent, action, world):
        """
        Predict the next blue state with a simplified one-step model that
        matches World.update_agent_state closely enough for greedy scoring.

        This forward model never mutates the real environment state.
        """
        action = np.asarray(action, dtype=float)

        g_eff = 1.0
        dt = getattr(world, "dt", 0.1)

        n_x = 2.0 * np.clip(action[0], -1.0, 1.0)
        n_z = 1.0 + 3.0 * np.clip(action[1], -1.0, 1.0)
        roll_cmd = (math.pi / 2.0) * np.clip(action[2], -1.0, 1.0)

        current_roll = getattr(agent.state, "roll", 0.0)
        current_pitch = getattr(agent.state, "pitch", 0.0)
        current_yaw = getattr(agent.state, "yaw", 0.0)
        current_v = getattr(agent.state, "p_vel", 1.0)

        roll_rate_limit = math.pi / 2.0
        roll_error = roll_cmd - current_roll
        roll_step = np.clip(roll_error, -roll_rate_limit * dt, roll_rate_limit * dt)

        next_roll = current_roll + roll_step
        next_roll = np.clip(next_roll, -math.pi / 2.0, math.pi / 2.0)

        v = max(current_v, 0.3)
        gamma = current_pitch
        phi = next_roll

        v_dot = g_eff * (n_x - math.sin(gamma))
        pitch_dot = (g_eff / v) * (n_z * math.cos(phi) - math.cos(gamma))

        cos_gamma = max(abs(math.cos(gamma)), 0.2)
        yaw_dot = (g_eff * n_z * math.sin(phi)) / (v * cos_gamma)

        # Keep the same blue-team maneuver disadvantage as the real dynamics.
        yaw_dot *= 0.8
        v_max = 2.5

        next_v = np.clip(current_v + v_dot * dt, 0.5, v_max)
        next_pitch = np.clip(current_pitch + pitch_dot * dt, -math.pi / 3, math.pi / 3)

        next_yaw = current_yaw + yaw_dot * dt
        next_yaw = self._wrap_angle(next_yaw)

        vx = next_v * math.cos(next_yaw) * math.cos(next_pitch)
        vy = next_v * math.sin(next_yaw) * math.cos(next_pitch)
        vz = next_v * math.sin(next_pitch)

        next_xy = agent.state.p_pos + np.array([vx * dt, vy * dt])
        next_z = agent.state.z_pos + vz * dt

        return next_xy, next_z, next_yaw, next_pitch, next_roll, next_v

    def _blue_guidance_candidates(self, agent, red_agents):
        """
        Build heuristic action candidates from the current target geometry.
        These are search candidates, not final actions.
        """
        candidates = []

        for target in red_agents:
            rel_p = target.state.p_pos - agent.state.p_pos
            target_yaw = math.atan2(rel_p[1], rel_p[0])
            yaw_diff = self._wrap_angle(target_yaw - agent.state.yaw)

            z_diff = target.state.z_pos - agent.state.z_pos

            roll_to_target = np.clip(yaw_diff / (math.pi / 2), -1.0, 1.0)
            nz_to_target = 0.5 if z_diff > 0 else -0.5

            # Direct pursuit.
            chase = np.array([0.8, nz_to_target, roll_to_target], dtype=np.float32)

            # More aggressive pursuit.
            aggressive_chase = np.array([1.0, nz_to_target, roll_to_target], dtype=np.float32)

            # Lateral flanking motion.
            flank_yaw = target_yaw + math.pi / 2
            flank_diff = self._wrap_angle(flank_yaw - agent.state.yaw)
            flank_roll = np.clip(flank_diff / (math.pi / 2), -1.0, 1.0)
            flank = np.array([0.8, 0.0, flank_roll], dtype=np.float32)

            # Reverse escape motion.
            evade_yaw = target_yaw + math.pi
            evade_diff = self._wrap_angle(evade_yaw - agent.state.yaw)
            evade_roll = np.clip(evade_diff / (math.pi / 2), -1.0, 1.0)
            evade = np.array([0.4, nz_to_target, evade_roll], dtype=np.float32)

            # Offset decoy motion.
            decoy_yaw = target_yaw + math.pi / 3
            decoy_diff = self._wrap_angle(decoy_yaw - agent.state.yaw)
            decoy_roll = np.clip(decoy_diff / (math.pi / 2), -1.0, 1.0)
            decoy = np.array([0.5, 0.0, decoy_roll], dtype=np.float32)

            candidates.extend([
                chase,
                aggressive_chase,
                flank,
                evade,
                decoy,
            ])

        return candidates

    def _blue_candidate_actions(self, agent, world, cfg, red_agents):
        """
        Construct the greedy-search candidate set for a blue agent.
        The set combines a small discrete action grid and tactic-biased samples.
        """
        candidates = []

        # Cover acceleration, climb/descent, and roll directions.
        speed_cmds = [0.3, 0.6, 0.9, 1.0]
        nz_cmds = [-0.6, 0.0, 0.6]
        roll_cmds = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for nx in speed_cmds:
            for nz in nz_cmds:
                for roll in roll_cmds:
                    candidates.append(np.array([nx, nz, roll], dtype=np.float32))

        # Add target-oriented heuristic actions to improve search quality.
        candidates.extend(self._blue_guidance_candidates(agent, red_agents))

        # Keep task diversity by biasing the candidate set with tactic-specific samples.
        tactic = getattr(agent, "tactic", 0)

        if tactic == 2:
            # Favor flanking turns.
            candidates.extend([
                np.array([0.8, 0.0, 1.0], dtype=np.float32),
                np.array([0.8, 0.0, -1.0], dtype=np.float32),
            ])

        elif tactic == 3:
            # High-altitude layer: bias toward climbing with attack-oriented turns.
            candidates.extend([
                np.array([0.6, 0.6, 1.0], dtype=np.float32),
                np.array([0.6, 0.6, -1.0], dtype=np.float32),
                np.array([0.8, 0.6, 0.5], dtype=np.float32),
                np.array([0.8, 0.6, -0.5], dtype=np.float32),
                np.array([0.6, 0.8, 0.0], dtype=np.float32),
            ])

        elif tactic == 4:
            # Low-altitude layer: bias toward descent with attack-oriented turns.
            candidates.extend([
                np.array([0.8, -0.6, 1.0], dtype=np.float32),
                np.array([0.8, -0.6, -1.0], dtype=np.float32),
                np.array([0.9, -0.6, 0.5], dtype=np.float32),
                np.array([0.9, -0.6, -0.5], dtype=np.float32),
                np.array([0.9, -0.8, 0.0], dtype=np.float32),
            ])

        elif tactic == 5:
            # Favor decoy or disturbance motion.
            candidates.extend([
                np.array([0.5, 0.0, 0.8], dtype=np.float32),
                np.array([0.5, 0.0, -0.8], dtype=np.float32),
                np.array([-0.2, 0.0, 0.8], dtype=np.float32),
                np.array([-0.2, 0.0, -0.8], dtype=np.float32),
            ])

        elif tactic == 6:
            # Favor defensive escape motion.
            candidates.extend([
                np.array([0.4, 0.5, 1.0], dtype=np.float32),
                np.array([0.4, 0.5, -1.0], dtype=np.float32),
                np.array([0.3, -0.5, 1.0], dtype=np.float32),
                np.array([0.3, -0.5, -1.0], dtype=np.float32),
            ])

        # Deduplicate candidates before scoring.
        unique_candidates = []
        seen = set()

        for action in candidates:
            action = np.clip(action, -1.0, 1.0)
            key = tuple(np.round(action, 3))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(action)

        return unique_candidates

    def _blue_tactic_weights(self, tactic):
        """
        Change greedy-scoring weights by tactic instead of hard-coding actions.
        This keeps the controller greedy while preserving task differences.
        """
        if tactic == 2:
            # Flank: emphasize heading geometry and lateral motion.
            return {
                "attack": 1.0,
                "angle": 1.2,
                "distance": 0.9,
                "threat": 1.0,
                "smooth": 1.0,
                "boundary": 1.0,
                "altitude": 0.0,
            }

        if tactic == 3:
            # High-altitude layer: value altitude control and angle quality.
            return {
                "attack": 1.0,
                "angle": 1.15,
                "distance": 0.85,
                "threat": 1.0,
                "smooth": 1.0,
                "boundary": 1.1,
                "altitude": 1.2,
            }

        if tactic == 4:
            # Low-altitude layer: value penetration and distance closing.
            return {
                "attack": 1.15,
                "angle": 1.0,
                "distance": 1.1,
                "threat": 1.1,
                "smooth": 1.0,
                "boundary": 1.1,
                "altitude": 1.2,
            }

        if tactic == 5:
            # Decoy: emphasize survival and disturbance.
            return {
                "attack": 0.85,
                "angle": 1.0,
                "distance": 0.8,
                "threat": 1.3,
                "smooth": 0.8,
                "boundary": 1.0,
                "altitude": 0.0,
            }

        if tactic == 6:
            # Defensive counter: strongly avoid red attack envelopes.
            return {
                "attack": 0.8,
                "angle": 0.9,
                "distance": 0.8,
                "threat": 1.8,
                "smooth": 1.0,
                "boundary": 1.0,
                "altitude": 0.0,
            }

        # Default offensive bias.
        return {
            "attack": 1.2,
            "angle": 1.0,
            "distance": 1.0,
            "threat": 1.0,
            "smooth": 1.0,
            "boundary": 1.0,
            "altitude": 0.0,
        }

    def _blue_greedy_score(self, agent, world, target, action, cfg):
        """
        Compute the one-step greedy utility J(a) for a blue action.
        Higher scores indicate better immediate utility under the current state.
        """
        next_xy, next_z, next_yaw, next_pitch, _, _ = self._predict_blue_next_state(
            agent,
            action,
            world
        )

        tactic = getattr(agent, "tactic", 0)
        weights = self._blue_tactic_weights(tactic)

        blue_attack_range = cfg.get("blue_attack_range", 3.5)
        blue_attack_angle = cfg.get("blue_attack_angle", math.pi / 6)

        red_attack_range = cfg.get("red_attack_range", 4.5)
        red_attack_angle = cfg.get("red_attack_angle", math.pi / 4)

        # Evaluate the predicted blue attack geometry on the chosen target.
        distance, ata = fast_compute_distance_and_angle_scalar(
            next_xy[0], next_xy[1], next_z,
            target.state.p_pos[0], target.state.p_pos[1], target.state.z_pos,
            next_yaw, next_pitch
        )

        in_attack_zone = distance < blue_attack_range and ata < blue_attack_angle

        # Entering the attack envelope gets a large immediate bonus.
        attack_score = 120.0 if in_attack_zone else 0.0

        # Low-health targets get a finishing bonus.
        target_hp = getattr(target, "hp", 100.0)
        if in_attack_zone and target_hp <= 20.0:
            attack_score += 40.0

        # Better nose-on-target geometry gets a higher score.
        angle_score = 10.0 * max(0.0, 1.0 - ata / math.pi)

        # Favor an effective attack distance instead of minimum distance.
        desired_dist = 0.75 * blue_attack_range
        distance_score = -2.0 * abs(distance - desired_dist)

        # Penalize future states that fall into red attack envelopes.
        threat_penalty = 0.0

        live_reds = [a for a in world.agents if a.team == 0 and not a.is_dead]
        for red in live_reds:
            red_dist, red_ata = fast_compute_distance_and_angle_scalar(
                red.state.p_pos[0], red.state.p_pos[1], red.state.z_pos,
                next_xy[0], next_xy[1], next_z,
                red.state.yaw, red.state.pitch
            )

            if red_dist < red_attack_range and red_ata < red_attack_angle:
                threat_penalty += 100.0
            else:
                # Apply a softer penalty when approaching the red attack zone.
                near_factor = max(
                    0.0,
                    (red_attack_range + 1.0 - red_dist) / (red_attack_range + 1.0)
                )
                angle_factor = max(
                    0.0,
                    (red_attack_angle + 0.5 - red_ata) / (red_attack_angle + 0.5)
                )
                threat_penalty += 20.0 * near_factor * angle_factor

        # Penalize hard boundary violations and risky altitude extremes.
        boundary_penalty = 0.0
        if abs(next_xy[0]) > 8.0 or abs(next_xy[1]) > 8.0 or next_z < 1.0 or next_z > 9.0:
            boundary_penalty += 80.0

        if next_z < 1.5 or next_z > 8.5:
            boundary_penalty += 20.0

        # Keep tactic 3/4 separated with altitude-layer preferences.
        altitude_score = 0.0
        if tactic == 3:
            desired_altitude = 7.2
            altitude_score = -abs(next_z - desired_altitude)
        elif tactic == 4:
            desired_altitude = 2.8
            altitude_score = -abs(next_z - desired_altitude)

        # Penalize large step-to-step action changes to reduce jitter.
        last_action = getattr(agent, "last_action", np.zeros(3))
        smooth_penalty = np.sum(np.square(action - last_action))

        score = (
            weights["attack"] * attack_score
            + weights["angle"] * angle_score
            + weights["distance"] * distance_score
            + weights["altitude"] * altitude_score
            - weights["threat"] * threat_penalty
            - weights["boundary"] * boundary_penalty
            - weights["smooth"] * 0.5 * smooth_penalty
        )

        return score

    def blue_action_callback(self, agent, world):
        """
        Greedy blue-team controller.

        Per step:
        1. Collect live red targets.
        2. Build a finite candidate action set.
        3. Predict one-step outcomes for each candidate.
        4. Evaluate J(a) for each candidate.
        5. Execute the argmax candidate.

        Usage:
        World.step calls this automatically through
        agent.action_callback(agent, world).
        """
        action = np.zeros(3, dtype=np.float32)

        red_agents = [a for a in world.agents if a.team == 0 and not a.is_dead]
        if agent.is_dead or not red_agents:
            return action

        cfg = getattr(world, "task_cfg", TASK_CONFIGS[META_TRAIN_TASK_IDS[0]])

        # Preserve the original task-driven tactic switching mechanism.
        self._assign_blue_tactics(world, cfg)

        # Record close combat entry without reverting the original flag behavior.
        my_pos = (agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos)
        dists = [
            math.sqrt(
                (r.state.p_pos[0] - my_pos[0]) ** 2
                + (r.state.p_pos[1] - my_pos[1]) ** 2
                + (r.state.z_pos - my_pos[2]) ** 2
            )
            for r in red_agents
        ]

        min_dist = min(dists)
        if min_dist <= 6.0:
            agent.combat_mode = True

        candidate_actions = self._blue_candidate_actions(
            agent=agent,
            world=world,
            cfg=cfg,
            red_agents=red_agents
        )

        best_action = None
        best_score = -1e18

        # Greedy selection: keep the action with the best immediate utility.
        for raw_action in candidate_actions:
            scaled_action = self._scale_blue_action(raw_action, cfg)

            action_score = max([
                self._blue_greedy_score(
                    agent=agent,
                    world=world,
                    target=target,
                    action=scaled_action,
                    cfg=cfg
                )
                for target in red_agents
            ])

            if action_score > best_score:
                best_score = action_score
                best_action = scaled_action

        if best_action is None:
            best_action = np.array([0.8, 0.0, 0.0], dtype=np.float32)

        # Keep the task-configured blue noise for disturbance-style tasks.
        noise_std = cfg.get("blue_noise", 0.15)
        best_action = best_action + np.random.normal(0, noise_std, size=3)

        return np.clip(best_action, -1.0, 1.0)

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
