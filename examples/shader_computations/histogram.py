def transform_input(x):
    if x == 0:
        return 1.0 / 256.0 - 1.0
    if x < 0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    else:
        x = (1.0 + x * 253.0) / 255.0
    return 2.0 * (x * 255.0 + 0.5) / 256.0 - 1.0


def main():
    input_values = [
        0.0,
        0.5,
        1.0,
        1.5,
        -1.5,
    ]

    for x in input_values:
        print(f"transform_input({x}) = {transform_input(x)}")


if __name__ == "__main__":
    main()
