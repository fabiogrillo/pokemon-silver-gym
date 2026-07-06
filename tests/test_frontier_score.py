"""
Unit tests for env.frontier_archive.frontier_score — waypoint-ordinal re-scoring (R4, technique
from docs/superpowers/specs/2026-07-06-final-attempt-findings.md §2).

Pure logic, NO PyBoy required. Validates:
  - delivered cells now rank by the CELL'S OWN max_waypoint ordinal (leading-edge sampling), so
    gym-adjacent cells outrank New Bark / Cherrygrove cells instead of being flat-tied at 1.0
  - the carry (2.0) and pre-egg (0.0) tiers — the agent_060/061 flat-tier fixes — are untouched

Run:
    .venv/bin/python -m tests.test_frontier_score
"""

from env.frontier_archive import frontier_score


def test_delivered_cells_ranked_by_waypoint():
    base = dict(egg_received=True, return_progress=0, gym=False)
    s0 = frontier_score(egg_delivered=True, max_waypoint=0, **base)
    s3 = frontier_score(egg_delivered=True, max_waypoint=3, **base)
    s5 = frontier_score(egg_delivered=True, max_waypoint=5, **base)
    assert s0 < s3 < s5


def test_carry_and_preegg_tiers_unchanged():
    assert frontier_score(True, False, 0, 0, False) == 2.0
    assert frontier_score(False, False, 0, 0, False) == 0.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"ALL {len(tests)} FRONTIER SCORE TESTS PASSED")


if __name__ == "__main__":
    main()
