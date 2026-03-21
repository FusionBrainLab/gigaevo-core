def entrypoint():
    """Return a max-subarray-sum solver function.

    The solver is deliberately naive (brute force O(n^2)) and slightly buggy
    on all-negative arrays to leave room for evolution to improve it.
    """

    def solve(nums: list[int]) -> int:
        n = len(nums)
        best = 0  # Bug: should be nums[0] for all-negative arrays
        for i in range(n):
            current = 0
            for j in range(i, n):
                current += nums[j]
                if current > best:
                    best = current
        return best

    return solve
