from pprint import pprint
import numpy as np


def try_compositing(
    num_steps,
    near_far_distance,
    alpha_values,
    colors,
    background_color,
    num_original_voxels,
):
    # brightness_factor = near_far_distance / (num_steps - 1)
    sample_distance = 1 / num_original_voxels
    our_spacing = 1 / num_steps
    actual_spacing = our_spacing / sample_distance
    print(f"actual_spacing: {actual_spacing}")
    established_color = background_color.copy()
    for i in range(num_steps):
        # alpha = alpha_values[i] * brightness_factor
        alpha = (1 - ((1 - alpha_values[i]) ** actual_spacing))
        previous_alpha = established_color[-1]
        color = colors[i]
        established_color[:3] = (
            established_color[:3] + (1 - previous_alpha) * color * alpha
        )
        established_color[-1] = previous_alpha + (1 - previous_alpha) * alpha
    
    print(f"alpha_values: {alpha}")

    return established_color


if __name__ == "__main__":
    # num_steps_to_try = [2, 4, 6]
    # num_steps_to_try = [16, 32, 64, 128, 256, 512, 1024]
    num_steps_to_try = [48, 52, 56, 60, 64, 68, 72, 76, 80]
    num_original_voxels = 64
    near_far_distance = 1.0
    background_color = np.array([0.0, 0.0, 0.0, 0.0])
    result_dict = {}
    for val in num_steps_to_try:
        alpha_values = np.full(shape=(val,), fill_value=0.1)
        color_values = np.array([np.array([0.1, 0.3, 0.2]) for _ in range(val)])
        result_dict[val] = try_compositing(
            val,
            near_far_distance,
            alpha_values,
            color_values,
            background_color,
            num_original_voxels,
        )

    pprint(result_dict)
