import numpy as np

class HiddenMarkovModel:

    def __init__(self, num_states, num_observations) -> None:
        
        # x: observations
        # y: states
        self.num_states = num_states
        self.num_observations = num_observations
        self.transition_matrix = np.zeros((num_states, num_states))
        self.emission_matrix = np.zeros((num_states, num_observations))
        self.initial_probabilities = np.zeros(num_states)

    def initialize_probabilities(self, transitions, emissions, initials):
        
        self.transition_matrix = transitions
        self.emission_matrix = emissions
        self.initial_probabilities = initials

    def likelihood(self, observations):
        return self._forward(observations)
    
    def decode(self, observations):
        return self._viterbi(observations)

    def fit(self, observations, max_iter=100):
        return self._baum_welch(observations, max_iter)

    def _forward(self, observations):

        A = self.transition_matrix
        B = self.emission_matrix
        pi = self.initial_probabilities
        o = observations # list of observation by index
        alpha = np.zeros(self.num_states)

        for t in range(len(observations)):

            if t == 0:
                # alpha_1(i) = pi_i * B_i(o_1)
                # element-wise multiplication
                alpha = pi * B[:, o[0]]
            else:
                # alpha_t(i) = sum(alpha_t-1(j) * A_ji) * B_i(o_t)
                # matrix form: alpha_t = (alpha_t-1 x A) * B[:, o_t]
                alpha = alpha @ A * B[:, o[t]]

        return alpha.sum()

    def _viterbi(self, observations):
        pass

    def _baum_welch(self, observations, max_iter):
        pass

    