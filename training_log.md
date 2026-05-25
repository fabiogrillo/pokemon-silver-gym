# Training Log — Pokemon Silver Gym (RL Agent)

Quick reference for all training strategies attempted, results, and lessons learned.
**Never propose a strategy already listed in "Failed / Abandoned" sections.**

---

## PPO_1 — First run (baseline)
- 10M steps, 8 envs
- Exploration: visited_tiles +1/new tile, visited_maps +100/new map (reset per episode), step penalty -0.01
- Events: rival, mr_pokemon, elm egg detected via full-byte comparison (BUG — compared full byte value, not bit edge)
- Sprout Tower floor flags at 0xD85C/0xD85D (unverified, later confirmed wrong)
- **Result**: reward_events = 0 (detection bug, not navigation). Addresses also wrong.

## PPO_2 — Fixed event detection, added revisited penalty
- 10M steps, 8 envs, ent_coef = 0.01
- Events: switched to bitwise edge detection (rising/falling bit per flag) — correct
- Exploration: added revisited tile penalty -0.01/step on top of step penalty -0.01/step
- **Result**: policy collapse (ep_rew_mean = -8.85). -0.02/step made all movement expensive → policy converged deterministically.
- **Lesson**: double step penalty kills entropy. Remove revisited penalty; keep only flat step penalty.

## PPO_3 — Removed revisited penalty, increased entropy
- 10M steps, 8 envs, ent_coef = 0.05
- Exploration: removed revisited tile penalty. visited_maps reset per episode (BUG). Map transition = +100.
- **Result**: reward hacking (ep_rew_mean = 515, visited_tiles = 139). Agent cycled ~9 local maps each episode for +100 each. value_loss = 22.6 (reward scale mismatch).
- **Lesson**: visited_maps must persist across episodes; map transition bonus must be one-shot lifetime, not per-episode.

## PPO_4 — Fixed hacking, reduced map bonus
- 10M steps, 8 envs, ent_coef = 0.05, gamma = 0.99, lr = 3e-4
- Exploration: visited_maps persists lifetime (no reset). Map transition = +20 one-shot.
- Events: rival +200, mr_pokemon +100, elm +200, badge +1000
- **Result**: genuine learning. ep_rew_mean = 242 (stable), visited_tiles = 305, value_loss = 0.381, explained_variance = 0.939. reward_events = 0 still — agent never navigated far enough to trigger events.
- **Lesson**: training mechanism is now correct. Core problem is credit assignment: Mr. Pokemon is 500+ steps away, gamma=0.99 → effective horizon ~100 steps.

---

## Verification — Event detection (2026-05-21)
- Ran `tests/test_event_detection.py` from `start.state` — full playthrough to Elm delivery
- **Mr. Pokemon egg (+100): CONFIRMED** — 0xD7BA bit6 rises at map(26,10)
- **Rival Cherrygrove (+200): CONFIRMED** — 0xD88E bit6 falls BEFORE battle starts (sprite visibility flag), then resets to 1 after leaving area. Detection is correct; the bit resets but the falling edge fires exactly once per encounter.
- **Elm delivery (+200): CONFIRMED** — 0xD7BA bit7 rises at map(24,5)
- **Map coordinates verified from log**: New Bark Town (24,5), Route 29 (24,3–4), Cherrygrove/Route 30 (26,3), Route 31 (26,1), Mr. Pokemon's house (26,10)
- **Sprout Tower 2F verified**: loaded `sprout_tower_2f.state` → map_bank=3, map_number=2. Code check `(3,2)` is correct.
- **Map Card (vecchio Cherrygrove)**: not in 0xD7B7–0xD8B6 event flag range. Skipped — entering Cherrygrove already gives +20 map transition.

## Save states — renamed for clarity (2026-05-21)
| File | Posizione | Uso curriculum |
|------|-----------|---------------|
| `start.state` | New Bark Town (24,4) | env 0-2 |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 3-4 |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 5-6 |
| `sprout_tower_2f.state` | Sprout Tower 2F (3,2) | env 7 |
| `after_mr_pokemon.state` | Route 31 area, uovo preso | non usato in curriculum |

---

## PPO_5 — Curriculum learning (2026-05-21)
- 30M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 3×start + 2×mid_route30 + 2×before_elm_delivery + 1×sprout_tower_2f
- New reward: Mr. Pokemon house (26,10) → +80 specific milestone (one-shot, on top of +20 map transition)
- **Result**: reward_events = 0.0122 (non-zero for first time!), ep_rew_mean = 221, visited_tiles = 240–277, value_loss stable 0.3–0.5 (final-batch spike to 22.2 is an artifact, not real), explained_variance = 0.9+ throughout.
- **Lesson**: curriculum DID cause reward_events to fire (envs 5-6 trigger Elm delivery immediately). Agent still not navigating consistently to events from start.state. sprout_tower_2f env creates value function instability without contributing to the main path toward Violet City.

---

## Verification — Violet City coordinates (2026-05-22)
- Ran `tests/test_violet_city_detection.py` from `mid_route30.state` — full playthrough to Zephyr Badge
- Full path confirmed: Route 30 → Route 29 → Elm delivery → Route 30 → Route 31 → Violet City → Sprout Tower (1F→2F→3F) → Pokemon Center → Gym → Falkner beaten
- **New map coordinates verified:**

| Luogo | map (bank, num) | Note |
|---|---|---|
| Route 30 building (Cherrygrove PC?) | (26,5) | breve visita da Route 30 |
| Route 30 building | (26,4) | breve visita da Route 30 |
| Violet City ovest (da Route 31) | **(26,2)** | prima mappa dopo Route 31 |
| Edificio in Violet City | (26,11) | breve visita, probabilmente una casa |
| Violet City principale | **(10,5)** | da qui: Torre Sprout, PC, Palestra |
| Pokemon Center Violet City | (10,10) | visita per cure |
| Sprout Tower 1F | (3,1) | ✓ già noto |
| Sprout Tower 2F | (3,2) | ✓ già noto |
| Sprout Tower 3F | (3,3) | ✓ già noto |
| **Violet City Gym (Falkner)** | **(10,7)** | badge ottenuto qui a tick=233720 |

- Rival encounter in Sprout Tower 3F = cutscene, nessun flag di battaglia separato da trackare
- 0xD7B8 bit0 rises subito dopo il badge → flag "Zephyr Badge ricevuto" (non usato nel training, solo osservazione)
- Total simulated reward: +1400 = 200 (rival) + 200 (elm) + 1000 (badge) ✓

## Save states — aggiornati (2026-05-22)
| File | Posizione | Uso curriculum |
|------|-----------|---------------|
| `start.state` | New Bark Town (24,4) | env 0-1 (PPO_7) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2-3 (PPO_7) |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (PPO_7) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (PPO_7) |
| `sprout_tower_2f.state` | Sprout Tower 2F (3,2) | rimosso dal curriculum |
| `after_mr_pokemon.state` | Route 31 area, uovo preso | non usato |

---

## PPO_6 — Violet City milestones + curriculum aggiustato (2026-05-22)
- 50M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 3×start + 2×mid_route30 + 3×before_elm_delivery
- New milestones in compute_reward: Violet City ovest (26,2) → +60, Violet City main (10,5) → +80, Violet City Gym (10,7) → +150
- **Result**: ep_rew_mean picco a ~350 a 15M steps, poi regressione a ~250 a fine run. reward_events = 0.0122 (solo Elm delivery dal curriculum, mai milestone Violet City). value_loss bimodale oscillante 0.5↔21 per tutta la run.
- **Lesson**: il curriculum con before_elm_delivery (reward immediato +200) vs start.state (reward sparse) crea distribuzioni di return troppo diverse — la value function oscilla tra i due regimi e destabilizza la policy gradient nella seconda metà del training. I milestone di Violet City non vengono mai raggiunti dagli env start.state perché il problema di credit assignment rimane irrisolto.

---

## PPO_7 — VecNormalize + violet_city curriculum (2026-05-22)
- 50M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city (distribuzione uniforme)
- **New**: VecNormalize(norm_obs=False, norm_reward=True) aggiunto in train.py — normalizza i reward su rolling mean/variance per env, eliminando il disallineamento di scala tra env types
- violet_city.state verificato: map=(10,5), HP=35/35 (curato al PC), party=5 pokemon, elm_delivery=done ✓
- **Result**: value_loss completamente stabile 0.007–0.116 (no più oscillazione bimodale). ep_rew_mean ~300, stabile per tutta la run. reward_events = 0.0122 (Elm delivery dai curriculum envs). explained_variance 0.9+ costante. Badge mai raggiunto.
- **Lesson**: VecNormalize risolve il bimodal value_loss. Il problema residuo è credit assignment verso la gym — gli env violet_city non raggiungono (10,7) in 50M steps.

---

## PPO_8 — Gym milestone aumentato + violet_city_gym curriculum (2026-05-23)
- 50M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 2×start + 2×mid_route30 + 2×before_elm_delivery + 1×violet_city + 1×violet_city_gym
- violet_city_gym.state: map=(10,7), local=(12,5), party=5, HP=45/45, elm_delivery=done ✓
- Gym milestone (10,7) aumentato da +150 a +400
- **Result**: value_loss stabile 0.007–0.07 (VecNormalize ancora efficace). ep_rew_mean 296-302. reward_events max 0.0244 nell'ultimo rollout (= 400/16384 = gym milestone sparato per la prima volta a ~50M steps). Nuovo segnale: 0.00305 (= 50/16384 = Pokemon catch). Badge mai raggiunto (reward_events non raggiunge ~0.061).
- **Lesson**: gym milestone ha sparato solo nell'ultimissimo rollout — il segnale è arrivato troppo tardi per propagarsi. L'env violet_city_gym.state non ha contribuito alla milestone perché la mappa è già in visited_maps all'init: il reward +400 può sparare solo per transizione entrante, non per l'env che parte già dentro. Serve battle win reward per incentivare a superare i trainer in palestra.

---

## Save states — aggiornati (2026-05-23)
| File | Posizione | Uso curriculum |
|------|-----------|---------------|
| `start.state` | New Bark Town (24,4) | env 0-1 (PPO_9) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2-3 (PPO_9) |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (PPO_9) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (PPO_9) |
| `violet_city_gym.state` | Violet City Gym (10,7), 2 passi dentro | rimosso dal curriculum (PPO_9) |

---

## PPO_9 — Battle win reward + 2×violet_city + 100M steps (2026-05-24)
- 100M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4, CHECKPOINT_FREQ=12_500_000
- Curriculum: 2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city (rimosso violet_city_gym)
- **New reward**: battle win +15 — `prev_battle_type > 0 and battle_type == 0 and hp_ratio > 0`
- **Result**: picco ep_rew_mean ~1500 a 60-70M steps (badge ottenuto dagli env violet_city!), poi regressione a ~954 a fine run. reward_events 0.073-0.076 (≈ 1 badge + ~20 battle wins per rollout). value_loss stabile 0.04-0.09. in_battle 0.26-0.29, confermando che il battle win reward funziona. Badge ottenuto ma policy fragile a fine training (entropy collapse).
- **Lesson**: il badge viene ottenuto da violet_city.state (partial win condition). Ma start.state non impara il percorso completo — il segnale del badge (+1000 a 500+ step di distanza) viene scontato a quasi 0 con gamma=0.99. La regressione finale (entropy collapse) suggerisce che la policy diventa troppo deterministica e il failure mode (morire ai trainer) non viene corretto. Servono: per-episode waypoints per il percorso da start.state, e gamma più alto per estendere l'orizzonte.

---

## PPO_10 — Per-episode route waypoints + gamma=0.995 (NEXT RUN)
- 100M steps, 8 envs, ent_coef=0.05, **gamma=0.995**, lr=3e-4
- Curriculum: invariato (2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city)
- **New**: `episode_maps` set, reset ogni episodio — rewards per route traversal che sparano una volta per episodio (non once per lifetime):
  - Route 31 (26,1): +20/episode
  - Violet City West (26,2): +30/episode (rimosso dal blocco one-shot, solo +20 one-shot generico rimane)
  - Violet City Main (10,5): +40/episode (rimosso dal blocco one-shot)
  - Gym (10,7): rimane +400 one-shot (prevenire hacking)
- Gamma 0.99→0.995: orizzonte effettivo da ~100 a ~200 step. 0.995^500 ≈ 0.08 vs 0.99^500 ≈ 0.007 — 10× più signal dal badge verso le decisioni di navigazione da start.state
- Expected: start.state env inizia a ricevere gradient signal dai waypoints di Violet City. Badge da start.state nella seconda metà del training.
- Success criteria: reward_events trending up da inizio training, ep_rew_mean stabile senza regressione, badge ottenuto da eval su start.state

---

## Strategies To Try (After PPO_10)

### If PPO_10 succeeds (badge obtained from start.state):
- Valutazione quantitativa: badge rate su 100 episodi da start.state
- Passare all'agente LLM (Phase 5 del plan) per il confronto

### If PPO_10 fails (waypoints raggiunti ma no badge da start):
- Aggiungere route_31.state come curriculum state bridge (Opzione C)
- Oppure aumentare gamma a 0.999

### Ruled out / do not retry
- Revisited tile penalty (PPO_2 lesson)
- visited_maps reset per episode (PPO_3 lesson)
- Map transition bonus > +30 per-lifetime (causes reward hacking — PPO_3 lesson; per-episode è ok se piccolo)
- Unverified Sprout Tower flags 0xD85C/0xD85D (wrong addresses)
- ent_coef < 0.05 with this reward scale (premature convergence)
- Map Card event reward (not in standard flag range, Cherrygrove +20 suffices)
- sprout_tower_2f in curriculum (isola narrativa, non contribuisce al path verso Violet City)
- 3×before_elm_delivery senza VecNormalize (troppa varianza di return, causa regressione — PPO_6 lesson)
- violet_city_gym.state nel curriculum (gym milestone non può sparare per quell'env — mappa già in visited_maps all'init — PPO_8 lesson)
- Battle reward proporzionale a HP (catena causale troppo lunga, doppia lettura RAM nello stesso step sempre 0)
