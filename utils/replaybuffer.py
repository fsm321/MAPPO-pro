import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.buffer_size, args.state_dim))
        self.share_s = np.zeros((args.buffer_size, args.share_state_dim))
        self.a = np.zeros((args.buffer_size, args.action_dim))
        # PPO stores the summed joint log-probability of the continuous action.
        self.a_logprob = np.zeros((args.buffer_size, 1))
        self.r = np.zeros((args.buffer_size, 1))
        self.s_ = np.zeros((args.buffer_size, args.state_dim))
        self.share_s_ = np.zeros((args.buffer_size, args.share_state_dim))
        self.dw = np.zeros((args.buffer_size, 1))
        self.done = np.zeros((args.buffer_size, 1))
        self.count = 0

    def store(self, s, share_s, a, a_logprob, r, s_, share_s_, dw, done):
        self.s[self.count] = s
        self.share_s[self.count] = share_s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.share_s_[self.count] = share_s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        n = self.count
        s = torch.tensor(self.s[:n], dtype=torch.float)
        share_s = torch.tensor(self.share_s[:n], dtype=torch.float)
        a = torch.tensor(self.a[:n], dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob[:n], dtype=torch.float)
        r = torch.tensor(self.r[:n], dtype=torch.float)
        s_ = torch.tensor(self.s_[:n], dtype=torch.float)
        share_s_ = torch.tensor(self.share_s_[:n], dtype=torch.float)
        dw = torch.tensor(self.dw[:n], dtype=torch.float)
        done = torch.tensor(self.done[:n], dtype=torch.float)

        return s, share_s, a, a_logprob, r, s_, share_s_, dw, done
