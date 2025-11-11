import numpy as np
import tensorstore as ts

DIR = r"/media/starfish/LargeSSD/metacell/data/calico/image/s0"

dataset = ts.open(
    {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": DIR},
    }
).result()

print(dataset[0:100, 0:100].read().result())
print(dataset.shape)
