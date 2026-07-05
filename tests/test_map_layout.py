from agents.rl import map_layout as ml


OVERWORLD = [(24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (10, 5)]


def test_tile_px_matches_image_scale():
    assert ml.TILE_PX == 16


def test_overworld_maps_not_inset_and_interiors_inset():
    for key in OVERWORLD:
        assert ml.MAP_INFO[key].inset is False
    for key in [(24, 5), (10, 7), (26, 11), (26, 10)]:
        assert ml.MAP_INFO[key].inset is True


def test_all_offsets_non_negative():
    for box in ml.MAP_INFO.values():
        assert box.offset[0] >= 0 and box.offset[1] >= 0


def test_route29_newbark_adjacency():
    """Route 29's east edge must touch New Bark's west edge (they are connected east/west)."""
    r29, nb = ml.MAP_INFO[(24, 3)], ml.MAP_INFO[(24, 4)]
    assert r29.offset[0] + r29.size[0] == nb.offset[0]


def test_route30_31_vertical_adjacency():
    """Route 31 sits directly north of Route 30 (south edge of 31 touches north edge of 30)."""
    r31, r30 = ml.MAP_INFO[(26, 2)], ml.MAP_INFO[(26, 1)]
    assert r31.offset[1] + r31.size[1] == r30.offset[1]


def test_to_image_px_uses_anchor_and_tile_px():
    box = ml.MAP_INFO[(24, 4)]
    px = ml.to_image_px(24, 4, 0, 0)
    assert px == (ml.ANCHOR_PX[0] + box.offset[0] * ml.TILE_PX,
                  ml.ANCHOR_PX[1] + box.offset[1] * ml.TILE_PX)


def test_unknown_map_returns_none():
    assert ml.to_image_px(99, 99, 0, 0) is None


def test_corridor_bbox_contains_all_overworld_maps():
    x0, y0, x1, y1 = ml.corridor_bbox_px()
    for key in OVERWORLD:
        box = ml.MAP_INFO[key]
        bx0 = ml.ANCHOR_PX[0] + box.offset[0] * ml.TILE_PX
        by0 = ml.ANCHOR_PX[1] + box.offset[1] * ml.TILE_PX
        assert x0 <= bx0 and y0 <= by0
        assert x1 >= bx0 + box.size[0] * ml.TILE_PX and y1 >= by0 + box.size[1] * ml.TILE_PX


def test_ram_to_image_px_unswaps_coordinates():
    assert ml.ram_to_image_px(24, 4, 3, 10) == ml.to_image_px(24, 4, 10, 3)
