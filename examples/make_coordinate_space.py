import json

old_space_str = r'{"x": [6.000000000000001e-9, "m"],"y": [6.000000000000001e-9,"m"], "z": [3.0000000000000004e-8, "m"]}'

old_space = json.loads(old_space_str)
print(old_space)

new_space_str = r'[{"name": "x", "scale": [6.000000000000001e-9, "m"]}, {"name": "y", "scale": [6.000000000000001e-9, "m"]}, {"name": "z", "scale": [3.0000000000000004e-8, "m"]}]'

new_space = json.loads(new_space_str)
print(new_space)

from neuroglancer import CoordinateSpace

old_coordinate_space = CoordinateSpace(old_space)
print(old_coordinate_space)

new_coordinate_space = CoordinateSpace(new_space)
print(new_coordinate_space)

co_ordinate_array_space = [
    {
        "name": "x",
        "coordinates": [0, 1, 2, 3, 4],
        "labels": ["a", "b", "c", "d", "e"],
    }
]

coordinate_space = CoordinateSpace(co_ordinate_array_space)
print(coordinate_space)
