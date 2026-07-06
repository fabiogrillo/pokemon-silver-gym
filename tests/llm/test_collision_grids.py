import json, os
import pytest
from agents.rl import map_layout as ml

GRIDS = {
    (24, 4): "assets/collision/new_bark.json", (24, 3): "assets/collision/route_29.json",
    (26, 3): "assets/collision/cherrygrove.json", (26, 1): "assets/collision/route_30.json",
    (26, 2): "assets/collision/route_31.json", (10, 5): "assets/collision/violet_city.json",
    (10, 7): "assets/collision/gym.json",
}


@pytest.mark.parametrize("key,path", GRIDS.items())
def test_grid_exists_and_matches_map_size(key, path):
    assert os.path.exists(path)
    g = json.load(open(path))
    assert (g["bank"], g["num"]) == key
    box = ml.MAP_INFO[key]
    assert g["width"] == box.size[0] and g["height"] == box.size[1]
    assert len(g["walkable"]) == g["height"] and len(g["walkable"][0]) == g["width"]
    flat = [c for row in g["walkable"] for c in row]
    assert 0 < sum(flat) < len(flat)  # some walkable, some not
