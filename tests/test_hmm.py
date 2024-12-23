import numpy as np
from model.hmm import HiddenMarkovModel

def test():
    model = HiddenMarkovModel(
        num_states=3,
        num_observations=3
    )

    model.initialize_probabilities(
        initials=None,
        transitions=np.array([
            [0.2, 0.5, 0.3],
            [0.6, 0.1, 0.3],
            [0.7, 0.1, 0.2]  
        ]),
        emissions=np.array([
            [0.3, 0.6, 0.1],
            [0.25, 0.05, 0.7],
            [0.85, 0.05, 0.1]
        ])
    )

    obs = np.array([2, 0, 0])
    print(f"Likelihood of observing {obs}: {model.likelihood(obs)}")
    print(f"Most likely state sequence: {model.decode(obs)}")
