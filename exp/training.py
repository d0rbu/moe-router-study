import math


def exponential_to_linear_save_steps(total_steps: int, save_every: int) -> set[int]:
    num_exponential_save_steps = math.ceil(math.log2(save_every))

    # exponential ramp and then linear
    # 0, 1, 2, 4, ..., save_every, save_every * 2, save_every * 3, ...
    save_steps = set(range(start=0, stop=total_steps, step=save_every))
    save_steps += {2**i for i in range(num_exponential_save_steps)}
    save_steps.add(0)

    return save_steps
