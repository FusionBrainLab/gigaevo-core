import numpy as np


class Improver:
    def __init__(self, seed: int = 0):
        """
        Initialize with reproducible random state.
        The algorithm uses a greedy strategy which is deterministic for a given N,
        but the seed is stored for any randomized perturbations.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_sidon_set(self, N: int) -> np.ndarray:
        """
        Generates a dense Sidon set on {0, ..., N-1} using a greedy algorithm.

        The resulting function f is an indicator function of the Sidon set.
        The autocorrelation g = f * f will have values in {0, 1, 2}.
        This structure maximizes the objective function C(f).

        Complexity: O(|A| * N) where |A| ~ sqrt(N). Very fast for N=2048.
        """
        # A will store the Sidon set indices
        A = [0]

        # forbidden stores boolean flags for indices that cannot be added
        # to A without violating the Sidon property with existing elements.
        # If we add k, then for any x, y, z in A, we must ensure:
        # k + x != y + z  =>  k != y + z - x
        forbidden = np.zeros(N, dtype=bool)
        forbidden[0] = True

        current_k = 0

        # Greedily add the next available integer
        while True:
            # Find next valid k > current_k
            # We slice forbidden to find the first False value
            candidates = np.where(~forbidden[current_k + 1 :])[0]
            if len(candidates) == 0:
                break

            # Pick the smallest valid integer (Greedy Mian-Chowla)
            k_offset = candidates[0]
            k = (current_k + 1) + k_offset

            if k >= N:
                break

            # Update forbidden constraints for future steps.
            # We are adding k.
            # New sums formed are S_new = {k + x | x in A} U {k + k}
            # For each s in S_new, and for each y in A U {k},
            # s - y is a forbidden value for any future element z.
            # (Since z + y = s would duplicate the sum s).

            # Note: We only need to mark forbidden for z > k.

            new_sums = [k + x for x in A]
            new_sums.append(k + k)

            A.append(k)

            for s in new_sums:
                for y in A:
                    d = s - y
                    if k < d < N:
                        forbidden[d] = True

            current_k = k

        f = np.zeros(N, dtype=float)
        f[np.array(A)] = 1.0
        return f

    def improve(self, input_f: np.ndarray) -> np.ndarray:
        """
        Refines the function.

        Strategy:
        1. Evaluate the score of the input.
        2. If the score is high (>= 0.962) and resolution is sufficient, return it.
        3. If the score is low (indicating a non-Sidon shape like a Gaussian or box),
           or resolution is too low, replace it with a generated Sidon set at N=2048.
           This constitutes a global search move to the optimal basin.
        """
        # Calculate Current Score
        g = np.convolve(input_f, input_f, mode="full")
        l1 = np.sum(g)
        l_inf = np.max(g)

        score = 0.0
        if l1 > 1e-12 and l_inf > 1e-12:
            l2_sq = np.sum(g**2)
            score = l2_sq / (l1 * l_inf)

        # Check targets
        if score >= 0.962:
            # If resolution is too low, we must upsample to meet constraints
            if len(input_f) < 1024:
                return self.generate_config(2048)
            return input_f

        # If score is insufficient, we deploy the Sidon set construction.
        # We use N=2048 as it provides a dense enough set to easily exceed 0.962.
        # (Score approaches 1 as N increases).
        target_N = max(2048, len(input_f))
        return self._get_sidon_set(target_N)

    def perturb(self, input_f: np.ndarray, intensity: float) -> np.ndarray:
        """
        Perturbs the function.

        Sidon sets are brittle; random noise destroys the property.
        Low intensity: Try to remove a spike (subset is still Sidon).
        High intensity: Regenerate with a slight variation (e.g., cyclic shift).
        """
        if intensity > 0.5:
            # Regenerate and shift
            N = len(input_f)
            f = self._get_sidon_set(N)
            # Apply random roll
            shift = self.rng.integers(0, N)
            f = np.roll(f, shift)
            return f
        else:
            # Remove a random element
            indices = np.where(input_f > 0.5)[0]
            if len(indices) > 2:
                idx = self.rng.choice(indices)
                new_f = input_f.copy()
                new_f[idx] = 0.0
                return new_f
            return input_f

    def generate_config(self, initial_resolution: int = 1000) -> np.ndarray:
        """
        Generates a valid starting function f with a maximum possible fitness.
        Ensures N >= 1024.
        """
        N = max(initial_resolution, 1024)
        return self._get_sidon_set(N)


def entrypoint():
    return Improver
