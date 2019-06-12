import numpy as np
import random
from models.lstm_burn_in.baselines_utils import SumSegmentTree, MinSegmentTree


class SeqReplayBuffer(object):
    """
    ある決まった長さ  < エピソード長さ
    の軌道を取り出し、それに関して最適化。切り取った系列の一番最後だけ割引収益にする必要がおそらくある。
    """

    def __init__(self, obs_dim, action_dim, max_replay_buffer_size, out_seq_length,
                 burn_in_steps, seed=0, positive_ratio=0.5):
        max_replay_buffer_size = int(max_replay_buffer_size)

        self._observation_dim = obs_dim
        self._action_dim = action_dim
        self._max_buffer_size = max_replay_buffer_size
        self.out_seq_length = out_seq_length
        self.seed = seed
        self.burn_in_steps = burn_in_steps
        self.positive_ratio = positive_ratio
        np.random.seed(seed)

        self._observations = []
        self._actions = []
        self._rewards = []
        self._lengths = []
        self._terminates = []
        self._lstm_states = []
        self._observations1 = []

        self._positive_observations = []
        self._positive_actions = []
        self._positive_rewards = []
        self._positive_lengths = []
        self._positive_terminates = []
        self._positive_lstm_states = []
        self._positive_observations1 = []

        # self._terminals[i] = a terminal was received at time i
        self._top = 0
        self._size = 0
        self._positive_top = 0
        self._positive_size = 0

    def add_path(self, obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, terminate_seq):
        original_length = len(action_seq)
        res = original_length - self.out_seq_length

        if res < 0:  # padding
            obs_seq = np.concatenate((obs_seq, np.zeros((-res,) + self._observation_dim)))
            action_seq = np.concatenate((action_seq, np.zeros(-res)))
            reward_seq = np.concatenate((reward_seq, np.zeros(-res)))
            terminate_seq = np.concatenate((terminate_seq, np.zeros(-res)))
            obs1_seq = np.concatenate((obs1_seq, np.zeros((-res,) + self._observation_dim)))

        if len(self._observations) < self._max_buffer_size:
            self._observations.append(np.array(obs_seq))
            self._actions.append(np.array(action_seq))
            self._rewards.append(np.array(reward_seq))
            self._lengths.append(original_length)
            self._terminates.append(np.array(terminate_seq))
            self._lstm_states.append(np.array(lstm_state_seq))
            self._observations1.append(np.array(obs1_seq))
        else:
            self._observations[self._top] = np.array(obs_seq)
            self._actions[self._top] = np.array(action_seq)
            self._rewards[self._top] = np.array(reward_seq)
            self._lengths[self._top] = original_length
            self._terminates[self._top] = np.array(terminate_seq)
            self._lstm_states[self._top] = np.array(lstm_state_seq)
            self._observations1[self._top] = np.array(obs1_seq)

        if (np.asarray(reward_seq) > 0).sum() > 0:
            if len(self._positive_observations) < self._max_buffer_size:
                self._positive_observations.append(np.array(obs_seq))
                self._positive_actions.append(np.array(action_seq))
                self._positive_rewards.append(np.array(reward_seq))
                self._positive_lengths.append(original_length)
                self._positive_terminates.append(np.array(terminate_seq))
                self._positive_lstm_states.append(np.array(lstm_state_seq))
                self._positive_observations1.append(np.array(obs1_seq))
            else:
                self._positive_observations[self._positive_top] = np.array(obs_seq)
                self._positive_actions[self._positive_top] = np.array(action_seq)
                self._positive_rewards[self._positive_top] = np.array(reward_seq)
                self._positive_lengths[self._positive_top] = original_length
                self._positive_terminates[self._positive_top] = np.array(terminate_seq)
                self._positive_lstm_states[self._positive_top] = np.array(lstm_state_seq)
                self._positive_observations1[self._positive_top] = np.array(obs1_seq)

            self._positive_advance()

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def _positive_advance(self):
        self._positive_top = (self._positive_top + 1) % self._max_buffer_size
        if self._positive_size < self._max_buffer_size:
            self._positive_size += 1

    def reformat(self, index, buffer_type="normal"):
        """
        padding や　収益計算の辻褄合わせなど
        :param index:
        :return:
        """
        if buffer_type == "positive":
            seq_length = self._positive_lengths[index]
            obs_seq = self._positive_observations[index]
            action_seq = self._positive_actions[index]
            reward_seq = self._positive_rewards[index]
            terminate_seq = self._positive_terminates[index]
            lstm_state = self._positive_lstm_states[index]
            obs1_seq = self._positive_observations1[index]

        else:
            seq_length = self._lengths[index]
            obs_seq = self._observations[index]
            action_seq = self._actions[index]
            reward_seq = self._rewards[index]
            terminate_seq = self._terminates[index]
            lstm_state = self._lstm_states[index]
            obs1_seq = self._observations1[index]

        res = seq_length - self.out_seq_length

        if res > 0:
            lim = seq_length - self.out_seq_length
            idx = np.random.randint(0, lim)

            start_burn_in_idx = max(idx - self.burn_in_steps, 0)
            lstm_state = lstm_state[start_burn_in_idx]
            burn_in_seq = obs_seq[start_burn_in_idx : idx]
            burn_in_length = len(burn_in_seq)
            if burn_in_length == 0:
                burn_in_seq = np.zeros((self.burn_in_steps,) + self._observation_dim)
            elif burn_in_length < self.burn_in_steps:
                burn_in_seq = np.concatenate((burn_in_seq, np.zeros((self.burn_in_steps - burn_in_length,) + self._observation_dim)))

            obs_seq = obs_seq[idx:idx + self.out_seq_length]
            action_seq = action_seq[idx:idx + self.out_seq_length]
            reward_seq = reward_seq[idx:idx + self.out_seq_length]
            terminate_seq = terminate_seq[idx:idx + self.out_seq_length]
            obs1_seq = obs1_seq[idx:idx + self.out_seq_length]
            seq_length = self.out_seq_length
            if len(burn_in_seq) != self.burn_in_steps:
               print("in res > 0 len :{}, idx: {}, seq_length: {}, burninsteps: {}, lim:{}".format(len(burn_in_seq), idx, self._lengths[index], self.burn_in_steps, lim))
           
        else:  # res == 0 (res > 0 does not occur for the padding of add_path)
            lstm_state = lstm_state[0]
            burn_in_seq = np.zeros((self.burn_in_steps,) + self._observation_dim)
            burn_in_length = 0
            if len(burn_in_seq) != self.burn_in_steps:
               print("in res == 0 len is :{}".format(len(burn_in_seq)))

        return [obs_seq, lstm_state, action_seq, reward_seq, obs1_seq, terminate_seq, seq_length, burn_in_seq, burn_in_length]

    def random_batch(self, batch_size, positive_ratio=None):
        """
        :param batch_size:
        :return:
        """
        ratio = positive_ratio if positive_ratio is not None else self.positive_ratio
        positive_batch = min(self._positive_size, int(batch_size * ratio))
        if positive_batch > 0:
            batch_size -= positive_batch
            positive_indicies = np.random.randint(0, self._positive_size, positive_batch)

        indices = np.random.randint(0, self._size, batch_size)

        samples = [self.reformat(i) for i in indices]
        if positive_batch > 0:
            tmp = [self.reformat(i, "positive") for i in positive_indicies]
            samples.extend(tmp)

        samples = list(zip(*samples))  # [[[1,2], [3,4,5]], [[6,7],[8,9,10]]] --> [([1,2], [6,7]), ([3,4,5], [8,9,10])]
        # print("switch:{}".format(np.mean(np.asarray(samples[3])[:, :, -2])))
        # return dict(
        #     obs_seqs=list(samples[0]),
        #     lstm_state=list(samples[1]),
        #     action_seqs=list(samples[2]),
        #     reward_seqs=list(samples[3]),
        #     obs1_seqs=list(samples[4]),
        #     terminate_seqs=list(samples[5]),
        #     seq_lengths=list(samples[6]),
        #     burn_in_seq=list(samples[7]),
        #     burn_in_length=list(samples[8]),
        # )
        return samples


class ReplayBuffer(object):
    def __init__(self, size, output_length, burn_in_length):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.output_l = output_length
        self.burn_l = burn_in_length

    def __len__(self):
        return len(self._storage)

    def add_path(self, obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq):
        obs_shape = np.array(obs_seq[0]).shape
        original_length = len(action_seq)
        res = original_length - self.output_l
        if res < 0:  # padding
            pad_obs = [obs1_seq[-1] for j in range(-res)]
            obs_seq.extend(pad_obs)
            action_seq.extend([action_seq[-1] for j in range(-res)])
            reward_seq.extend([0. for j in range(-res)])
            done_seq.extend(1. for j in range(-res))
            obs1_seq.extend(pad_obs)

        n_seqment = int(len(action_seq) / self.output_l)  # compare length (after padding)
        for i in range(n_seqment):
            start = i * self.output_l
            end = start + self.output_l
            seq_length = original_length if res < 0 else self.output_l

            start_burn = max(start - self.burn_l, 0)
            burn_in_seq = np.array(obs_seq[start_burn:start])
            burn_in_length = len(burn_in_seq)
            if burn_in_length == 0:
                burn_in_seq = np.zeros((self.burn_l,) + obs_shape)
            elif burn_in_length < self.burn_l:
                burn_in_seq = np.concatenate((burn_in_seq, np.zeros((self.burn_l - burn_in_length,) + obs_shape)))

            self.add(obs_seq[start:end], lstm_state_seq[start_burn], action_seq[start:end], reward_seq[start:end],
                     obs1_seq[start:end], done_seq[start:end], seq_length, burn_in_seq, burn_in_length)

        # same process to above
        overrap = 0 if len(action_seq) % self.output_l == 0 else 1
        if overrap == 1:
            start = len(action_seq) - self.output_l
            seq_length = self.output_l

            start_burn = max(start - self.burn_l, 0)
            burn_in_seq = np.array(obs_seq[start_burn:start])
            burn_in_length = len(burn_in_seq)
            if burn_in_length == 0:
                burn_in_seq = np.zeros((self.burn_l,) + obs_shape)
            elif burn_in_length < self.burn_l:
                burn_in_seq = np.concatenate((burn_in_seq, np.zeros((self.burn_l - burn_in_length,) + obs_shape)))
            self.add(obs_seq[start:], lstm_state_seq[start_burn], action_seq[start:], reward_seq[start:],
                     obs1_seq[start:], done_seq[start:], seq_length, burn_in_seq, burn_in_length)

    def add(self, obs_seq, lstm_state, action_seq, reward_seq, obs1_seq, done_seq, seq_length, burn_in_seq, burn_in_length):
        data = (obs_seq, lstm_state, action_seq, reward_seq, obs1_seq, done_seq, seq_length, burn_in_seq, burn_in_length)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_seq, lstm_states, actions_seq, rewards_seq, obses1_seq, dones_seq, seq_lengths, burn_in_seqs, burn_in_lengths = [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_seq, lstm_state, action_seq, reward_seq, obs1_seq, done_seq, seq_length, burn_in_seq, burn_in_length = data
            obses_seq.append(np.array(obs_seq, copy=False))
            lstm_states.append(np.array(lstm_state, copy=False))
            actions_seq.append(np.array(action_seq, copy=False))
            rewards_seq.append(reward_seq)
            obses1_seq.append(np.array(obs1_seq, copy=False))
            dones_seq.append(done_seq)
            seq_lengths.append(seq_length)
            burn_in_seqs.append(np.array(burn_in_seq, copy=False))
            burn_in_lengths.append(burn_in_length)

        return np.array(obses_seq), np.array(lstm_states),np.array(actions_seq), np.array(rewards_seq), \
               np.array(obses1_seq), np.array(dones_seq), np.array(seq_lengths), np.array(burn_in_seqs), np.array(burn_in_lengths)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, output_length, burn_in_length):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, output_length, burn_in_length)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

class PositiveReplayBuffer(object):
    def __init__(self, replaybuffer0, replaybuffer1):
        self.rb_total = replaybuffer0
        self.rb_positive = replaybuffer1
        self.positive_ratio = 0.5

    def add_path(self, obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq):
        self.rb_total.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq)
        if (np.asarray(reward_seq) > 0).sum() > 0:
            self.rb_positive.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq)

    def sample(self, batch_size, beta=0.4):
        if len(self.rb_positive) < 2:
            return tuple(list(self.rb_total.sample(batch_size, beta)) + [0])
        else:
            size_positive = int(self.positive_ratio * batch_size)
            size_total = batch_size - size_positive
            sample_positive = self.rb_positive.sample(size_positive, beta)
            sample_total = self.rb_total.sample(size_total, beta)
            output = [np.concatenate((sample_positive[i], sample_total[i]), axis=0) for i in range(len(sample_positive))]
            assert len(output[0]) == batch_size
            return tuple(list(output) + [size_positive])

    def update_ratio(self, positive_ratio):
        self.positive_ratio = positive_ratio

    def __len__(self):
        return len(self.rb_total)

    def update_priorities(self, idxes, positive_size, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # assert len(idxes) == len(priorities)
        if positive_size == 0:
            self.rb_total.update_priorities(idxes, priorities)
        else:
            self.rb_positive.update_priorities(idxes[:positive_size], priorities[:positive_size])
            self.rb_total.update_priorities(idxes[positive_size:], priorities[positive_size:])


if __name__ == '__main__':
    buffer_size = 10
    max_length = 4
    burn_in_steps = 2
    rb0 = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
    rb1 = PrioritizedReplayBuffer(size=buffer_size, alpha=0.9, output_length=max_length, burn_in_length=burn_in_steps)
    rb = PositiveReplayBuffer(rb0, rb1)

    obs_seq = np.ones(max_length)
    lstm_state_seq = np.ones(max_length)
    action_seq = np.ones(max_length)
    reward_seq = np.zeros(max_length)
    reward_seq1 = np.ones(max_length)
    obs1_seq = np.ones(max_length)
    done_seq = np.zeros(max_length)
    rb.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq)
    rb.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq1, obs1_seq, done_seq)
    rb.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq, obs1_seq, done_seq)
    rb.add_path(obs_seq, lstm_state_seq, action_seq, reward_seq1, obs1_seq, done_seq)

    _ = rb.sample(4, 0.4)
    pass