import json

old_space_str = r'{"x": [6.000000000000001e-9, "m"],"y": [6.000000000000001e-9,"m"], "z": [3.0000000000000004e-8, "m"]}'

old_space = json.loads(old_space_str)
print(old_space)

new_space_str = r'[{"name": "x", "scale": [6.000000000000001e-9, "m"]}, {"name": "y", "scale": [6.000000000000001e-9, "m"]}, {"name": "z", "scale": [3.0000000000000004e-8, "m"]}]'

new_space = json.loads(new_space_str)
print(new_space)

from neuroglancer import CoordinateSpace, url_state

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

my_full_url = r"https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.000000000000001e-9%2C%22m%22%5D%2C%22y%22:%5B6.000000000000001e-9%2C%22m%22%5D%2C%22z%22:%5B3.0000000000000004e-8%2C%22m%22%5D%7D%2C%22position%22:%5B5523.99072265625%2C8538.9384765625%2C1198.0423583984375%5D%2C%22crossSectionScale%22:3.7621853549999242%2C%22projectionOrientation%22:%5B-0.0040475670248270035%2C-0.9566215872764587%2C-0.22688281536102295%2C-0.18271005153656006%5D%2C%22projectionScale%22:4699.372698097029%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected%22%2C%22tab%22:%22source%22%2C%22name%22:%22corrected-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth%22%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.63%2C%22notSelectedAlpha%22:0.14%2C%22segments%22:%5B%223208%22%2C%224901%22%2C%2213%22%2C%224965%22%2C%224651%22%2C%222282%22%2C%223189%22%2C%223758%22%2C%2215%22%2C%224027%22%2C%223228%22%2C%22444%22%2C%223207%22%2C%223224%22%2C%223710%22%5D%2C%22name%22:%22ground_truth%22%7D%5D%2C%22layout%22:%224panel%22%7D"

state = url_state.parse_url(my_full_url)
state_as_json = state.to_json()
state_as_json["dimensions"] = CoordinateSpace(state_as_json["dimensions"]).to_json()
with open("state.json", "w") as f:
    json.dump(
        state_as_json,
        f,
        ensure_ascii=False,
    )
