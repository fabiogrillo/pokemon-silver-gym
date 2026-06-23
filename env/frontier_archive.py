"""
Go-Explore / frontier-reset archive (agent_059).

A process-shared, on-disk archive of "frontier cells" — PyBoy save-states harvested from the
START POLICY'S OWN trajectory. A fraction of env resets load from this archive instead of
start.state, so deep sub-skills (the Elm egg-delivery A-press; later the Falkner fight) are
PRACTICED IN-DISTRIBUTION, from states the policy actually visits — with no foreign save-state,
hence no visual-island segregation (the project's central, repeatedly-proven failure mode).

Why the filesystem IS the data structure: SubprocVecEnv runs each env in its own process with no
shared memory. Storing each cell as one file in a shared directory lets every worker read/write the
same archive with zero IPC — a frontier discovered by one worker seeds resets in all the others.
`sample()` re-lists the directory each call (sub-ms for a few thousand files) so the view is always
fresh and crash-safe. Atomic writes (tmp + os.replace) keep concurrent writers from tearing a file.

File format:  {score:09d}__{cellkey}.state   (raw PyBoy save_state bytes)
  - score   : int frontier rank, zero-padded for lexical sort (see frontier_score)
  - cellkey : digits+single-underscores (see cell_key); the '__' double-underscore is the delimiter
"""

import os
import glob
import random

from .rewards import MR_POKEMON_BIT, ELM_BIT


def cell_key(ram_state, k=4):
    """Coarse identity of a frontier position. Buckets coords by k tiles AND splits on the egg/gym
    story flags, so 'Cherrygrove WITH the undelivered egg' is a distinct cell from 'Cherrygrove
    without it' — exactly the state the delivery backtrack must be practiced from."""
    bank = ram_state["map_bank"]
    num = ram_state["map_number"]
    bx = ram_state["local_x"] // k
    by = ram_state["local_y"] // k
    er = 1 if ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT else 0
    ed = 1 if ram_state["flag_elm_mr_pokemon"] & ELM_BIT else 0
    gym = int(ram_state.get("gym_trainers_beaten", 0))
    return f"{bank}_{num}_{bx}_{by}_{er}_{ed}_{gym}"


def frontier_score(egg_received, egg_delivered, return_progress, max_waypoint, gym):
    """FLAT priority TIER from the cell's OWN story flags — NOT its depth (agent_061 fix).

    agent_060 used depth-scoring (100 + return_progress), where return_progress is the EPISODE max.
    When a deep episode passed through a physically-shallow carry cell, add()'s "higher score
    replaces" rule overwrote that shallow cell's score to a deep value → the archive ended up with
    163 lab-with-egg cells (105) but ~0 Cherrygrove-with-egg cells (102). Sampling then almost never
    reset at the shallow carry states where the START episodes actually are, so the "push south from
    Cherrygrove" segment was never practiced and delivery never transferred (nav/egg_delivered flat 0).

    Flat tiers fix both failure modes: (1) a cell's score never changes, so add() keeps the FIRST
    captured (shallow-included) state per cell — shallow carry cells persist; (2) all carry cells are
    EQUAL, so ε-greedy samples uniformly across depths → shallow AND deep carry states get practiced.
    carry (2) > delivered (1) > pre-egg (0) keeps Phase-1 transfer from being starved by Phase-2 cells.
    (return_progress/max_waypoint/gym kept in the signature for telemetry/future Phase-2 reprioritizing.)"""
    if egg_delivered:
        return 1.0   # Phase-2 (post-delivery) — kept, but deprioritized vs the Phase-1 backtrack
    if egg_received:
        return 2.0   # carry (egg in hand, undelivered) — the Phase-1 practice we want, FLAT across depth
    return 0.0       # pre-egg (start policy already does this)


class FrontierArchive:
    """On-disk, process-shared cell→state archive. See module docstring."""

    def __init__(self, root_dir, max_cells=4000, epsilon=0.1, rng=None):
        self.root = root_dir
        self.max_cells = max_cells
        self.epsilon = epsilon          # uniform-sampling floor (diversity vs frontier exploitation)
        self.rng = rng or random.Random()
        os.makedirs(self.root, exist_ok=True)

    # ── internal: parse the directory (the source of truth, re-read each call) ──────────────────
    def _list(self):
        """Return [(score:int, cellkey:str, path:str)] for every cell file currently on disk."""
        out = []
        for path in glob.glob(os.path.join(self.root, "*.state")):
            name = os.path.basename(path)[:-len(".state")]
            score_str, _, cellkey = name.partition("__")
            try:
                out.append((int(score_str), cellkey, path))
            except ValueError:
                continue  # ignore malformed / partial files
        return out

    def __len__(self):
        return len(self._list())

    def _find(self, cellkey):
        """(score, path) for an existing file of this cell, or None."""
        for path in glob.glob(os.path.join(self.root, f"*__{cellkey}.state")):
            name = os.path.basename(path)[:-len(".state")]
            score_str = name.partition("__")[0]
            try:
                return int(score_str), path
            except ValueError:
                continue
        return None

    def add(self, cellkey, score, get_bytes):
        """Record this cell if it is new or reached at a HIGHER frontier score. `get_bytes` is a
        zero-arg callable producing the save-state bytes — called ONLY when we will actually write,
        so the 200KB serialization is skipped for the common already-archived case.
        Returns True iff a file was written."""
        score = int(round(score))
        existing = self._find(cellkey)
        if existing is not None and existing[0] >= score:
            return False
        data = get_bytes()
        final = os.path.join(self.root, f"{score:09d}__{cellkey}.state")
        tmp = final + f".tmp.{os.getpid()}.{self.rng.randrange(1 << 30)}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, final)                       # atomic publish
        if existing is not None and existing[1] != final:
            try:
                os.remove(existing[1])               # drop the lower-score file for this cell
            except FileNotFoundError:
                pass
        self._evict()
        return True

    def _evict(self):
        """Bound the archive: while over capacity, drop the lowest-frontier cells."""
        cells = self._list()
        if len(cells) <= self.max_cells:
            return
        cells.sort(key=lambda c: c[0])               # ascending score
        for _, _, path in cells[: len(cells) - self.max_cells]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def sample(self):
        """Return the save-state bytes of one cell, or None if the archive is empty.
        ε-greedy: with prob epsilon pick uniformly (diversity); else pick proportional to frontier
        score (exploit promising cells). A race-evicted file is skipped on a re-list retry."""
        for _ in range(2):
            cells = self._list()
            if not cells:
                return None
            if self.rng.random() < self.epsilon:
                chosen = self.rng.choice(cells)
            else:
                weights = [max(s, 0) for s, _, _ in cells]
                total = sum(weights)
                chosen = (self.rng.choices(cells, weights=weights, k=1)[0]
                          if total > 0 else self.rng.choice(cells))
            try:
                with open(chosen[2], "rb") as f:
                    return f.read()
            except FileNotFoundError:
                continue                              # evicted under us → re-list and retry once
        return None
