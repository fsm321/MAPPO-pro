import torch
import torch.nn as nn
from algorithms.mappo import MAPPO_Continuous


class Meta_MAPPO_Continuous(MAPPO_Continuous):
    def __init__(self, args):
        super(Meta_MAPPO_Continuous, self).__init__(args)
        self.initial_entropy = args.entropy_coef

    def get_weights(self):
        # Clone actor/critic weights on the current device.
        # This avoids repeated CPU-GPU copies during each meta step.
        actor_weights = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
        critic_weights = {k: v.detach().clone() for k, v in self.critic.state_dict().items()}
        return actor_weights, critic_weights

    def meta_update(self, old_weights, meta_lr):
        old_actor, old_critic = old_weights
        with torch.no_grad():
            # Reptile-style first-order meta update for the actor.
            # Move the initialization toward the post-adaptation parameters.
            for name, param in self.actor.named_parameters():
                if name in old_actor:
                    param.data = old_actor[name] + meta_lr * (param.data - old_actor[name])

            # Apply the same first-order interpolation to the centralized critic.
            for name, param in self.critic.named_parameters():
                if name in old_critic:
                    param.data = old_critic[name] + meta_lr * (param.data - old_critic[name])

    def meta_train_step(
            self,
            support_buffer,
            query_buffer,
            total_steps,
            meta_lr,
            support_group_size,
            query_group_size,
            inner_epochs=1,
            outer_epochs=1
    ):
        """
        Reptile-style support/query meta step:
        1. Save the pre-adaptation parameters.
        2. Adapt on the support buffer with PPO updates.
        3. Continue updating on the query buffer.
        4. Move the saved initialization toward the post-query parameters.
        """
        old_weights = self.get_weights()

        # Support updates play the role of task-specific adaptation.
        support_actor_loss, support_critic_loss = self.update(
            support_buffer,
            total_steps,
            do_lr_decay=False,
            rollout_group_size=support_group_size,
            K_epochs_override=inner_epochs
        )

        query_actor_loss, query_critic_loss = self.update(
            query_buffer,
            total_steps,
            do_lr_decay=False,
            rollout_group_size=query_group_size,
            K_epochs_override=outer_epochs
        )

        self.meta_update(old_weights, meta_lr)

        # Decay learning rates once per meta step, not once per support/query sub-update.
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        return (
            support_actor_loss,
            support_critic_loss,
            query_actor_loss,
            query_critic_loss
        )

    def lr_decay(self, total_steps):
        # Linearly decay actor/critic learning rates and the entropy coefficient.
        progress = max(0.0, 1 - total_steps / self.max_train_steps)
        lr_a_now, lr_c_now = self.lr_a * progress, self.lr_c * progress

        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

        # Keep a small entropy floor so exploration does not collapse too early.
        self.entropy_coef = max(0.001, self.initial_entropy * progress)
