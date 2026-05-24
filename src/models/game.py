import numpy as np


class GroupRecommendationGame:
    def __init__(self, eta1=1.0, eta2=1.0, n_param=2.0):
        self.eta1 = eta1
        self.eta2 = eta2
        self.n_param = n_param

    def calculate_utility(self, strategy_set, local_i, uid, I, S):
        si = strategy_set[local_i]
        count_si = strategy_set.count(si)

        profit = I[uid, si] / count_si
        cost = self.eta1 * (S[uid, si] + 1 - I[uid, si]) ** self.n_param
        return self.eta2 * (profit - cost)

    def _build_payoff_tensor(self, group_user_indices, num_topics, I, S):
        N = len(group_user_indices)
        T = num_topics

        idx = np.indices([T] * N)

        payoffs = np.zeros([T] * N + [N])

        for i, uid in enumerate(group_user_indices):
            si = idx[i]                                        
            count_si = np.sum(idx == si[np.newaxis], axis=0)  

            profit = I[uid, si] / count_si
            cost = self.eta1 * (S[uid, si] + 1 - I[uid, si]) ** self.n_param
            payoffs[..., i] = self.eta2 * (profit - cost)

        return payoffs

    def find_nash_equilibrium(self, group_user_indices, num_topics, I, S, max_iter=50):
        payoffs = self._build_payoff_tensor(group_user_indices, num_topics, I, S)
        N = len(group_user_indices)

        # A profile is a pure NE iff every player is at their best response.
        # np.max over axis i gives the best payoff player i can achieve for each
        # combination of the other players' strategies.
        is_ne = np.ones([num_topics] * N, dtype=bool)
        for i in range(N):
            best = np.max(payoffs[..., i], axis=i, keepdims=True)
            is_ne &= (payoffs[..., i] >= best - 1e-10)

        ne_profiles = np.argwhere(is_ne)
        if len(ne_profiles) > 0:
            return ne_profiles[0].tolist()

        return self._best_response_fallback(group_user_indices, num_topics, I, S, max_iter)

    def _best_response_fallback(self, group_user_indices, num_topics, I, S, max_iter=50):
        num_players = len(group_user_indices)
        current_strategies = [np.argmax(I[uid]) for uid in group_user_indices]

        for _ in range(max_iter):
            changed = False
            for i in range(num_players):
                uid = group_user_indices[i]
                best_utility = self.calculate_utility(current_strategies, i, uid, I, S)
                best_topic = current_strategies[i]

                for topic_idx in range(num_topics):
                    if topic_idx == current_strategies[i]:
                        continue
                    temp_strategies = list(current_strategies)
                    temp_strategies[i] = topic_idx
                    new_utility = self.calculate_utility(temp_strategies, i, uid, I, S)
                    if new_utility > best_utility:
                        best_utility = new_utility
                        best_topic = topic_idx
                        changed = True

                current_strategies[i] = best_topic

            if not changed:
                break

        return current_strategies

    def recommend_items(self, group_user_indices, num_topics, I, S):
        return self.find_nash_equilibrium(group_user_indices, num_topics, I, S)
