from pprint import pprint
import numpy as np

def weight(alpha):
    a = 8.0 * alpha + 0.01
    b = -0.95 * 1.0 + 1.0
    return a * a * a * b * b * b

def try_compositing(
    num_steps,
    alpha_values,
    colors,
    background_color,
    num_original_voxels,
):
    sample_distance = 1 / num_original_voxels
    our_spacing = 1 / num_steps
    actual_spacing = our_spacing / sample_distance
    print(f"sampling ratio: {actual_spacing}")
    established_color = background_color.copy()
    summed = np.array([0.0, 0.0, 0.0, 0.0])
    alpha_product = 1.0
    for i in range(num_steps):
        alpha = alpha_values[i] * actual_spacing
        w = weight(alpha)
        summed[:3] += w * colors[i] * alpha
        summed[-1] += w * alpha
        alpha_product *= (1 - alpha)
    established_color[:3] = (summed[:3] / summed[-1]) * (1 - alpha_product) + background_color[:3] * alpha_product
    established_color[-1] = 1 - alpha_product

    return established_color


if __name__ == "__main__":
    # num_steps_to_try = [2, 4, 6]
    num_steps_to_try = [16, 32, 64, 128, 256, 512, 1024]
    # num_steps_to_try = [48, 52, 56, 60, 64, 68, 72, 76, 80]
    num_original_voxels = 64
    background_color = np.array([0.0, 0.0, 0.0, 0.0])
    result_dict = {}
    for val in num_steps_to_try:
        alpha_values = np.full(shape=(val,), fill_value=0.1)
        color_values = np.array([np.array([0.1, 0.3, 0.5]) for _ in range(val)])
        result_dict[val] = try_compositing(
            val,
            alpha_values,
            color_values,
            background_color,
            num_original_voxels,
        )

    pprint(result_dict)
