import os
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # Output the pre-squash Gaussian mean. The final action squashing is
        # handled outside the actor so log-prob correction can be computed.
        mean = self.mean_layer(s)
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.share_state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.v_layer = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.v_layer, gain=1.0)

    def forward(self, share_s):
        share_s = self.activate_func(self.fc1(share_s))
        share_s = self.activate_func(self.fc2(share_s))
        v_s = self.v_layer(share_s)
        return v_s


class MAPPO_Continuous:
    def __init__(self, args):
        self.device = args.device
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.n_red = getattr(args, "n_red", 2)
        self.num_envs = getattr(args, "num_envs", 1)
        self.rollout_group_size = self.n_red * self.num_envs
        self.max_train_steps = args.max_train_steps
        self.max_episode_steps = args.max_episode_steps  # Used by checkpoint naming and rollout truncation logic.
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.save_dir, self.date, self.model_dir = args.save_dir, args.date, args.model_dir

        self.actor = Actor_Gaussian(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr_a,
            eps=1e-5 if self.set_adam_eps else 1e-8
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_c,
            eps=1e-5 if self.set_adam_eps else 1e-8
        )

    def choose_action(self, s):
        s_tensor = torch.tensor(s, dtype=torch.float).to(self.device)
        is_single = len(s_tensor.shape) == 1
        if is_single:
            s_tensor = s_tensor.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_dist(s_tensor)
            raw_action = dist.rsample()
            a, a_logprob = self._squash_action_and_logprob(dist, raw_action)

        if is_single:
            return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()
        return a.cpu().numpy(), a_logprob.cpu().numpy()

    def choose_action_deterministic(self, s):
        s_tensor = torch.tensor(s, dtype=torch.float).to(self.device)
        is_single = len(s_tensor.shape) == 1
        if is_single:
            s_tensor = s_tensor.unsqueeze(0)

        with torch.no_grad():
            raw_mean = self.actor(s_tensor)
            a = self.max_action * torch.tanh(raw_mean)

        if is_single:
            return a.cpu().numpy().flatten()
        return a.cpu().numpy()

    def _squash_action_and_logprob(self, dist, raw_action):
        tanh_action = torch.tanh(raw_action)
        action = self.max_action * tanh_action
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(self.max_action * (1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def _inverse_squash_action(self, action):
        action_scaled = torch.clamp(action / self.max_action, -0.999999, 0.999999)
        return 0.5 * torch.log((1 + action_scaled) / (1 - action_scaled))

    def update(
            self,
            replay_buffer,
            total_steps,
            do_lr_decay=True,
            rollout_group_size=None,
            K_epochs_override=None
    ):
        s, share_s, a, a_logprob, r, s_next, share_s_next, dw, done, active_mask = replay_buffer.numpy_to_tensor()
        s = s.to(self.device)
        share_s = share_s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_next = s_next.to(self.device)
        share_s_next = share_s_next.to(self.device)
        dw = dw.to(self.device)
        done = done.to(self.device)
        active_mask = active_mask.to(self.device)
        old_group_size = self.rollout_group_size
        # Meta-MAPPO splits rollouts into support/query buffers, so GAE grouping
        # may need a temporary rollout size override during each update call.
        old_group_size = self.rollout_group_size
        if rollout_group_size is not None:
            self.rollout_group_size = rollout_group_size
        adv, v_target = self.get_adv(share_s, r, share_s_next, dw, done, active_mask)
        self.rollout_group_size = old_group_size
        valid_mask = active_mask.squeeze(-1) > 0.5
        valid_count = int(valid_mask.sum().item())

        if valid_count <= 0:
            return 0.0, 0.0

        # Prioritized sampling only draws valid transitions.
        adv_abs = torch.abs(adv).squeeze(-1)
        adv_abs = adv_abs * active_mask.squeeze(-1)
        sample_prob = torch.zeros_like(adv_abs)

        if bool(torch.isnan(adv_abs).any().item()) or adv_abs.sum().item() <= 1e-8:
            sample_prob[valid_mask] = 1.0 / valid_count
        else:
            priority_prob = adv_abs[valid_mask] + 1e-6
            priority_prob = priority_prob / priority_prob.sum()

            uniform_prob = torch.ones_like(priority_prob) / valid_count
            mixed_prob = 0.7 * priority_prob + 0.3 * uniform_prob
            sample_prob[valid_mask] = mixed_prob

        a_loss_sum, c_loss_sum = 0, 0
        K_epochs = self.K_epochs if K_epochs_override is None else K_epochs_override
        buffer_size = s.shape[0]
        mini_batch_size = min(self.mini_batch_size, valid_count)
        effective_batch_size = min(self.batch_size, valid_count)
        batch_count = max(1, effective_batch_size // mini_batch_size)

        for _ in range(K_epochs):
            for _ in range(batch_count):
                index = torch.multinomial(sample_prob, mini_batch_size, replacement=False)
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(dim=-1, keepdim=True)
                raw_action = self._inverse_squash_action(a[index])
                _, a_logprob_now = self._squash_action_and_logprob(dist_now, raw_action)
                ratio = torch.exp(a_logprob_now - a_logprob[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                a_loss_each = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                mb_active = active_mask[index]
                a_loss = (a_loss_each * mb_active).sum() / (mb_active.sum() + 1e-8)

                self.optimizer_actor.zero_grad()
                a_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(share_s[index])
                c_loss_each = F.smooth_l1_loss(v_s, v_target[index], reduction='none')
                c_loss = (c_loss_each * mb_active).sum() / (mb_active.sum() + 1e-8)
                self.optimizer_critic.zero_grad()
                c_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                a_loss_sum += a_loss.mean().item()
                c_loss_sum += c_loss.item()

        if do_lr_decay and self.use_lr_decay:
            self.lr_decay(total_steps)
        denom = K_epochs * batch_count
        return a_loss_sum / denom, c_loss_sum / denom

    # Compute GAE per rollout slot so trajectories from different agents or
    # parallel environments are not mixed together.
    def get_adv(self, share_s, r, share_s_next, dw, done, active_mask=None):
        with torch.no_grad():
            v_s = self.critic(share_s)
            v_s_next = self.critic(share_s_next)
            # dw marks true terminals, so bootstrap only when dw == 0.
            deltas = r + self.gamma * v_s_next * (1.0 - dw) - v_s
            group_size = self.rollout_group_size
            total_size = deltas.shape[0]

            # If the buffer cannot be reshaped into aligned rollout groups,
            # fall back to the flat reverse scan instead of crashing.
            if total_size % group_size != 0:
                adv = torch.zeros_like(deltas).to(self.device)
                gae = torch.zeros(1, 1, device=self.device)

                for t in reversed(range(total_size)):
                    gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - done[t])
                    adv[t] = gae

                v_target = adv + v_s
                if self.use_adv_norm:
                    if active_mask is not None:
                        valid = active_mask > 0.5
                        if valid.sum().item() > 1:
                            adv_mean = adv[valid].mean()
                            adv_std = adv[valid].std(unbiased=False)
                            adv = torch.where(
                                valid,
                                (adv - adv_mean) / (adv_std + 1e-5),
                                torch.zeros_like(adv)
                            )
                        else:
                            adv = torch.zeros_like(adv)
                    else:
                        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-5)

                return adv, v_target

            T = total_size // group_size

            # [T, group_size, 1]
            deltas = deltas.view(T, group_size, 1)
            done = done.view(T, group_size, 1)

            adv = torch.zeros_like(deltas).to(self.device)

            # Keep one GAE accumulator per rollout slot.
            gae = torch.zeros(group_size, 1, device=self.device)

            for t in reversed(range(T)):
                gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - done[t])
                adv[t] = gae

            # Flatten back to [batch_size, 1].
            adv = adv.view(-1, 1)
            v_target = adv + v_s

            if self.use_adv_norm:
                if active_mask is not None:
                    valid = active_mask > 0.5
                    if valid.sum().item() > 1:
                        adv_mean = adv[valid].mean()
                        adv_std = adv[valid].std(unbiased=False)
                        adv = torch.where(
                            valid,
                            (adv - adv_mean) / (adv_std + 1e-5),
                            torch.zeros_like(adv)
                        )
                    else:
                        adv = torch.zeros_like(adv)
                else:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-5)
        return adv, v_target

    def lr_decay(self, total_steps):
        progress = max(0.0, 1 - total_steps / self.max_train_steps)
        lr_a_now, lr_c_now = self.lr_a * progress, self.lr_c * progress
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, agent_id, total_num_steps):
        # Use self.max_episode_steps so save indices stay aligned with the current training setup.
        path = f"{self.save_dir}/{self.date}/model/{int(total_num_steps // self.max_episode_steps)}"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/actor_shared.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_shared.pt")

    def restore(self, agent_id):
        # Load checkpoints onto the current device for cross-device compatibility.
        self.actor.load_state_dict(torch.load(f"{self.model_dir}/actor_shared.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{self.model_dir}/critic_shared.pt", map_location=self.device))
