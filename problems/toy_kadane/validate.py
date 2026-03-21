"""Validate max subarray sum solutions against test cases."""


def _reference(nums: list[int]) -> int:
    """Kadane's algorithm — the correct answer."""
    max_sum = curr_sum = nums[0]
    for x in nums[1:]:
        curr_sum = max(x, curr_sum + x)
        max_sum = max(max_sum, curr_sum)
    return max_sum


TEST_CASES = [
    [-2, 1, -3, 4, -1, 2, 1, -5, 4],
    [1],
    [5, 4, -1, 7, 8],
    [-1, -2, -3],
    [-1],
    [0],
    [1, 2, 3, 4, 5],
    [-3, 1, -2, 5, -1, 2],
    [100, -1, 100],
    [-5, 4, 6, -3, 4, -1],
    [2, -1, 2, 3, 4, -5],
    [1, -1, 1, -1, 1],
    [-2, -3, 4, -1, -2, 1, 5, -3],
    [3, -2, 5, -1],
    [-1, 3, -1, 3, -1],
    list(range(-50, 50)),
    [10000, -9999] * 100,
    [-1] * 100 + [500] + [-1] * 100,
    list(range(1, 101)),
    [(-1) ** i * i for i in range(1, 51)],
]


def validate(data):
    """Score a max-subarray-sum implementation against test cases.

    Args:
        data: the return value of entrypoint(nums) called on each test case.
              Actually, the framework calls entrypoint once — so we need to
              structure this differently. The framework calls:
                result = entrypoint(nums)
              for a SINGLE input. But we want to test many inputs.

              GigaEvo calls entrypoint() with no args, so we'll have the
              entrypoint define the function and we'll test it here.
    """
    # data is the return of entrypoint() — which should be a callable
    if callable(data):
        solve = data
    else:
        return {"is_valid": 0.0, "fitness": 0.0}

    correct = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        try:
            result = solve(case[:])  # pass a copy
            expected = _reference(case)
            if result == expected:
                correct += 1
        except Exception:
            pass

    fitness = correct / total

    return {
        "is_valid": 1.0 if fitness > 0 else 0.0,
        "fitness": fitness,
    }
