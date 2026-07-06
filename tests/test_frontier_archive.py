"""
Unit tests for env.frontier_archive — the Go-Explore / frontier-reset archive (agent_059).

Pure logic, NO PyBoy required (state payloads are fake bytes). Validates:
  - cell_key buckets coords and SPLITS on story flags (egg-in-hand vs not is a distinct cell)
  - frontier_score ranks: pre-egg < egg-carrying < egg-delivered/forward; gym depth ranks up
  - FrontierArchive.add writes a cell; re-adding the same cell does not duplicate; higher score replaces
  - sample() returns stored bytes, weighted toward higher-frontier cells; None when empty
  - eviction keeps the highest-frontier cells when over max_cells
  - CROSS-PROCESS sharing: a second archive on the same dir sees the first's cells (the SubprocVecEnv crux)

Run:
    .venv/bin/python -m tests.test_frontier_archive
"""

import random
import tempfile
from env.frontier_archive import FrontierArchive, cell_key, frontier_score

# Mirror the reward bitmasks so the test is independent of import wiring.
MR_POKEMON_BIT = 0x40
ELM_BIT = 0x80


def _ram(bank, num, x, y, egg_received=False, egg_delivered=False, gym=0):
    flag = (MR_POKEMON_BIT if egg_received else 0) | (ELM_BIT if egg_delivered else 0)
    return {
        "map_bank": bank, "map_number": num,
        "local_x": x, "local_y": y,
        "flag_elm_mr_pokemon": flag,
        "gym_trainers_beaten": gym,
    }


def test_cell_key_buckets_coords():
    # Same K-bucket → same cell; crossing the bucket boundary → different cell.
    a = cell_key(_ram(24, 3, 0, 0), k=4)
    b = cell_key(_ram(24, 3, 3, 3), k=4)   # same 4x4 bucket
    c = cell_key(_ram(24, 3, 4, 0), k=4)   # next bucket in x
    assert a == b, f"coords in same bucket must share a key: {a} vs {b}"
    assert a != c, f"coords across a bucket boundary must differ: {a} vs {c}"


def test_cell_key_splits_on_story_flags():
    # "Cherrygrove WITH egg" must be a DIFFERENT cell from "Cherrygrove without egg".
    no_egg = cell_key(_ram(26, 3, 5, 5, egg_received=False))
    with_egg = cell_key(_ram(26, 3, 5, 5, egg_received=True))
    delivered = cell_key(_ram(26, 3, 5, 5, egg_received=True, egg_delivered=True))
    assert len({no_egg, with_egg, delivered}) == 3, "story flags must partition cells"


def test_frontier_score_flat_tiers_no_depth_bias():
    # agent_061 fix: score depends ONLY on the cell's own egg/delivered flags — NOT on depth.
    # 060 used 100+return_progress, so when a DEEP episode passed through a physically-shallow cell,
    # add()'s "higher score replaces" overwrote that shallow cell's score to a deep value → the
    # shallow Cherrygrove-with-egg states (where start episodes actually are) were destroyed and
    # never practiced → delivery never transferred to start (nav/egg_delivered flat 0 for 24M).
    # Flat tiers keep every carry cell EQUAL so shallow & deep carry states get uniform practice;
    # carry ranks ABOVE delivered so Phase-1 transfer isn't starved by Phase-2 cells.
    pre_egg = frontier_score(egg_received=False, egg_delivered=False, return_progress=0, max_waypoint=1, gym=0)
    carry_shallow = frontier_score(egg_received=True, egg_delivered=False, return_progress=1, max_waypoint=1, gym=0)
    carry_deep = frontier_score(egg_received=True, egg_delivered=False, return_progress=5, max_waypoint=1, gym=0)
    delivered = frontier_score(egg_received=True, egg_delivered=True, return_progress=5, max_waypoint=2, gym=0)
    delivered_gym = frontier_score(egg_received=True, egg_delivered=True, return_progress=5, max_waypoint=5, gym=1)
    assert carry_shallow == carry_deep, f"carry score must NOT depend on depth: {carry_shallow} vs {carry_deep}"
    # R4 (final-attempt findings §2): delivered cells are no longer flat — they rank by the CELL's
    # own waypoint ordinal so ε-greedy resets concentrate at the corridor's leading edge. Flatness
    # is preserved one level down: two delivered cells at the SAME waypoint are still equal.
    assert delivered < delivered_gym, "delivered cells rank by waypoint ordinal (R4 leading-edge sampling)"
    same_wp = frontier_score(egg_received=True, egg_delivered=True, return_progress=9, max_waypoint=2, gym=0)
    assert delivered == same_wp, "delivered cells at the SAME waypoint stay flat (no depth bias within a tier)"
    assert pre_egg < carry_shallow, f"bad tiers: pre={pre_egg} carry={carry_shallow}"
    assert pre_egg < delivered, "delivered always outranks pre-egg"


def test_add_and_sample_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        arc = FrontierArchive(d, max_cells=10, rng=random.Random(0))
        assert arc.sample() is None, "empty archive must sample None"
        wrote = arc.add("c1", 100, lambda: b"STATE_ONE")
        assert wrote is True and len(arc) == 1
        assert arc.sample() == b"STATE_ONE"


def test_add_same_cell_no_duplicate_lazy_bytes():
    with tempfile.TemporaryDirectory() as d:
        arc = FrontierArchive(d, max_cells=10, rng=random.Random(0))
        arc.add("c1", 100, lambda: b"FIRST")
        calls = {"n": 0}
        def get_bytes():
            calls["n"] += 1
            return b"SECOND"
        wrote = arc.add("c1", 100, get_bytes)   # same key, not higher score
        assert wrote is False, "re-adding same cell at equal score must not write"
        assert len(arc) == 1
        assert calls["n"] == 0, "must NOT serialize state bytes when it won't write (lazy)"
        assert arc.sample() == b"FIRST"


def test_higher_score_replaces():
    with tempfile.TemporaryDirectory() as d:
        arc = FrontierArchive(d, max_cells=10, rng=random.Random(0))
        arc.add("c1", 100, lambda: b"OLD")
        wrote = arc.add("c1", 105, lambda: b"NEW")   # deeper frontier for the SAME cell
        assert wrote is True and len(arc) == 1
        assert arc.sample() == b"NEW"


def test_sampling_weights_toward_frontier():
    with tempfile.TemporaryDirectory() as d:
        arc = FrontierArchive(d, max_cells=10, rng=random.Random(1234))
        arc.add("low", 1, lambda: b"LOW")
        arc.add("high", 1000, lambda: b"HIGH")
        draws = [arc.sample() for _ in range(400)]
        highs = draws.count(b"HIGH")
        assert highs > 300, f"high-frontier cell should dominate sampling, got {highs}/400"
        assert draws.count(b"LOW") > 0, "low cell should still be reachable (not starved to zero)"


def test_eviction_keeps_highest_frontier():
    with tempfile.TemporaryDirectory() as d:
        arc = FrontierArchive(d, max_cells=2, rng=random.Random(0))
        arc.add("a", 10, lambda: b"A")
        arc.add("b", 20, lambda: b"B")
        arc.add("c", 30, lambda: b"C")   # over cap → lowest (a) must be evicted
        assert len(arc) == 2
        seen = {arc.sample() for _ in range(200)}
        assert b"A" not in seen, "lowest-frontier cell must be evicted"
        assert b"B" in seen and b"C" in seen


def test_cross_process_sharing_via_filesystem():
    # Simulates two SubprocVecEnv workers: writer and reader share ONLY the directory.
    with tempfile.TemporaryDirectory() as d:
        writer = FrontierArchive(d, max_cells=10, rng=random.Random(0))
        reader = FrontierArchive(d, max_cells=10, rng=random.Random(0))
        writer.add("shared", 100, lambda: b"FROM_WRITER")
        assert len(reader) == 1, "reader must see the writer's cell via the shared dir"
        assert reader.sample() == b"FROM_WRITER"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"ALL {len(tests)} FRONTIER ARCHIVE TESTS PASSED")


if __name__ == "__main__":
    main()
