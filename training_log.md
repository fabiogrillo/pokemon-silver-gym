# Training Log — Pokemon Silver Gym (RL Agent)

Quick reference for all training strategies attempted, results, and lessons learned.
**Never propose a strategy already listed in "Failed / Abandoned" sections.**

## Agent naming convention (since 2026-06-12)

Every training run is an **Agent NNN** with a progressive number (PWhiddy-style). Historical runs
were renamed BOTH in this log (headers `Agent NNN (ex OLD)` + body references) AND on disk:
`runs/checkpoints/agent_0NN/` (zip prefixes included) and TensorBoard dirs `runs/agent_0NN[_1]/`.
ONE exception: the live Agent 045 still writes to `runs/PPO_CNN_10m_1` and
`runs/checkpoints/PPO_CNN_10m/` — rename those when it stops. New runs use `RUN_NAME="agent_046"`+.

Legacy-name bridge (for old notes, terminal history, external references):

| Agent | Legacy name | Agent | Legacy name | Agent | Legacy name |
|---|---|---|---|---|---|
| 001–018 | `PPO_1`…`PPO_18` (MLP) | 029 | `PPO_CNN_8_crossing_wp` | 038 | `PPO_CNN_10f` |
| 019–025 | `PPO_CNN_1`…`PPO_CNN_7` | 030 | `PPO_CNN_9_selfstate` | 039 | `PPO_CNN_10g` |
| 026 | `PPO_CNN_8_stage1` | 031 | `PPO_CNN_9_gymtest` | 040 | `PPO_CNN_10h` |
| 027 | `PPO_CNN_8_stage1b` | 032 | `PPO_CNN_10_smoke` | 041 | `PPO_CNN_10i` |
| 028 | `PPO_CNN_8_crossing` | 033 | `PPO_CNN_10` | 042 | `PPO_CNN_10j` |
| 034 | `PPO_CNN_10b` | 035 | `PPO_CNN_10c` | 043 | `PPO_CNN_10k` |
| 036 | `PPO_CNN_10d` | 037 | `PPO_CNN_10e` | 044 | `PPO_CNN_10L` |
| | | | | 045 | `PPO_CNN_10m` (live) |

---

## Agent 001 (ex PPO_1) — First run (baseline)
- 10M steps, 8 envs
- Exploration: visited_tiles +1/new tile, visited_maps +100/new map (reset per episode), step penalty -0.01
- Events: rival, mr_pokemon, elm egg detected via full-byte comparison (BUG — compared full byte value, not bit edge)
- Sprout Tower floor flags at 0xD85C/0xD85D (unverified, later confirmed wrong)
- **Result**: reward_events = 0 (detection bug, not navigation). Addresses also wrong.

## Agent 002 (ex PPO_2) — Fixed event detection, added revisited penalty
- 10M steps, 8 envs, ent_coef = 0.01
- Events: switched to bitwise edge detection (rising/falling bit per flag) — correct
- Exploration: added revisited tile penalty -0.01/step on top of step penalty -0.01/step
- **Result**: policy collapse (ep_rew_mean = -8.85). -0.02/step made all movement expensive → policy converged deterministically.
- **Lesson**: double step penalty kills entropy. Remove revisited penalty; keep only flat step penalty.

## Agent 003 (ex PPO_3) — Removed revisited penalty, increased entropy
- 10M steps, 8 envs, ent_coef = 0.05
- Exploration: removed revisited tile penalty. visited_maps reset per episode (BUG). Map transition = +100.
- **Result**: reward hacking (ep_rew_mean = 515, visited_tiles = 139). Agent cycled ~9 local maps each episode for +100 each. value_loss = 22.6 (reward scale mismatch).
- **Lesson**: visited_maps must persist across episodes; map transition bonus must be one-shot lifetime, not per-episode.

## Agent 004 (ex PPO_4) — Fixed hacking, reduced map bonus
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

## Agent 005 (ex PPO_5) — Curriculum learning (2026-05-21)
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
| `start.state` | New Bark Town (24,4) | env 0-1 (Agent 007) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2-3 (Agent 007) |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (Agent 007) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (Agent 007) |
| `sprout_tower_2f.state` | Sprout Tower 2F (3,2) | rimosso dal curriculum |
| `after_mr_pokemon.state` | Route 31 area, uovo preso | non usato |

---

## Agent 006 (ex PPO_6) — Violet City milestones + curriculum aggiustato (2026-05-22)
- 50M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 3×start + 2×mid_route30 + 3×before_elm_delivery
- New milestones in compute_reward: Violet City ovest (26,2) → +60, Violet City main (10,5) → +80, Violet City Gym (10,7) → +150
- **Result**: ep_rew_mean picco a ~350 a 15M steps, poi regressione a ~250 a fine run. reward_events = 0.0122 (solo Elm delivery dal curriculum, mai milestone Violet City). value_loss bimodale oscillante 0.5↔21 per tutta la run.
- **Lesson**: il curriculum con before_elm_delivery (reward immediato +200) vs start.state (reward sparse) crea distribuzioni di return troppo diverse — la value function oscilla tra i due regimi e destabilizza la policy gradient nella seconda metà del training. I milestone di Violet City non vengono mai raggiunti dagli env start.state perché il problema di credit assignment rimane irrisolto.

---

## Agent 007 (ex PPO_7) — VecNormalize + violet_city curriculum (2026-05-22)
- 50M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4 (invariati)
- Curriculum: 2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city (distribuzione uniforme)
- **New**: VecNormalize(norm_obs=False, norm_reward=True) aggiunto in train.py — normalizza i reward su rolling mean/variance per env, eliminando il disallineamento di scala tra env types
- violet_city.state verificato: map=(10,5), HP=35/35 (curato al PC), party=5 pokemon, elm_delivery=done ✓
- **Result**: value_loss completamente stabile 0.007–0.116 (no più oscillazione bimodale). ep_rew_mean ~300, stabile per tutta la run. reward_events = 0.0122 (Elm delivery dai curriculum envs). explained_variance 0.9+ costante. Badge mai raggiunto.
- **Lesson**: VecNormalize risolve il bimodal value_loss. Il problema residuo è credit assignment verso la gym — gli env violet_city non raggiungono (10,7) in 50M steps.

---

## Agent 008 (ex PPO_8) — Gym milestone aumentato + violet_city_gym curriculum (2026-05-23)
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
| `start.state` | New Bark Town (24,4) | env 0-1 (Agent 009) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2-3 (Agent 009) |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (Agent 009) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (Agent 009) |
| `violet_city_gym.state` | Violet City Gym (10,7), 2 passi dentro | rimosso dal curriculum (Agent 009) |

---

## Agent 009 (ex PPO_9) — Battle win reward + 2×violet_city + 100M steps (2026-05-24)
- 100M steps, 8 envs, ent_coef=0.05, gamma=0.99, lr=3e-4, CHECKPOINT_FREQ=12_500_000
- Curriculum: 2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city (rimosso violet_city_gym)
- **New reward**: battle win +15 — `prev_battle_type > 0 and battle_type == 0 and hp_ratio > 0`
- **Result**: picco ep_rew_mean ~1500 a 60-70M steps (badge ottenuto dagli env violet_city!), poi regressione a ~954 a fine run. reward_events 0.073-0.076 (≈ 1 badge + ~20 battle wins per rollout). value_loss stabile 0.04-0.09. in_battle 0.26-0.29, confermando che il battle win reward funziona. Badge ottenuto ma policy fragile a fine training (entropy collapse).
- **Lesson**: il badge viene ottenuto da violet_city.state (partial win condition). Ma start.state non impara il percorso completo — il segnale del badge (+1000 a 500+ step di distanza) viene scontato a quasi 0 con gamma=0.99. La regressione finale (entropy collapse) suggerisce che la policy diventa troppo deterministica e il failure mode (morire ai trainer) non viene corretto. Servono: per-episode waypoints per il percorso da start.state, e gamma più alto per estendere l'orizzonte.

---

## Agent 010 (ex PPO_10) — Per-episode route waypoints + gamma=0.995 (2026-05-24)
- 100M steps, 8 envs, ent_coef=0.05, **gamma=0.995**, lr=3e-4
- Curriculum: invariato (2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city)
- **New**: `episode_maps` set, reset ogni episodio — per-episode waypoints:
  - Cherrygrove (26,3): +25/episode
  - Route 31 (26,1): +50/episode
  - Violet City West (26,2): +80/episode
  - Violet City Main (10,5): +100/episode
  - Gym (10,7): +200/episode + rimane +400 one-shot
- Battle win reward (+15) ancora presente da Agent 009 — non rimosso
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+983, avg tiles=101 — battle grinding confermato: agent macina wild battles vicino a New Bark Town invece di navigare
- **Lesson**: battle win reward crea local optima devastante: +15/vittoria near start è più prevedibile dei waypoint distanti. Tiles=101 è la firma del grinding (pochi tile, molti step in battaglia). Rimuovere completamente il battle win reward.

---

## Agent 011 (ex PPO_11) — Removed battle win + gamma=0.999 + ent_coef=0.08 (2026-05-24 → 2026-05-26)
- 100M steps, 8 envs, **ent_coef=0.08** (era 0.05), **gamma=0.999**, lr=3e-4
- Curriculum: invariato (2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city)
- **Rimosso**: battle win reward (commentato in compute_reward)
- Per-episode waypoints invariati da Agent 010
- **Result**: ep_rew_mean=280–284 (stabile, no regressione). reward_events=0.0108 smoothed (solo Elm delivery dai curriculum envs). value_loss=0.0079 (migliore di sempre). entropy=-2.04 stabile per tutta la run. in_battle=0.075 (grinding eliminato). visited_tiles smoothed=155.
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+162.5±43.8, avg tiles=314.3±32.1, avg steps=16384 (tutti troncati, nessun episodio termina). Reward ≈ tiles − step_penalty: nessun waypoint event in quasi tutti gli episodi.
- **Lesson**: infrastruttura perfetta (no collapse, no grinding, value_loss record), ma credit assignment irrisolto. L'agente esplora ~314 tile ma non naviga direzionalmente verso Violet City. I per-episode waypoints (+25/+50/+80/+100) non sono abbastanza forti da trainare la policy a percorrere 800+ step in modo consistente. Serve un curriculum bridge in Route 31.

---

## Save states — aggiornati (2026-05-26)
| File | Posizione | Uso curriculum |
|------|-----------|---------------|
| `start.state` | New Bark Town (24,4) | env 0-1 (Agent 012) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2 (Agent 012) |
| `route_31.state` | Route 31 (26,1) | env 3 (Agent 012) — nuovo bridge |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (Agent 012) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (Agent 012) |
| `violet_city_gym.state` | Violet City Gym (10,7), 2 passi dentro | rimosso dal curriculum |

---

## Agent 012 (ex PPO_12) — route_31.state bridge curriculum (2026-05-26 → 2026-05-27)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- **Curriculum**: 2×start + 1×mid_route30 + **1×route_31** + 2×before_elm_delivery + 2×violet_city
- route_31.state: salvato manualmente il 2026-05-26 da save_state.py, mappa (26,1) verificata
- Reward invariato rispetto a Agent 011
- **Result**: ep_rew_mean finale 231–238 (declino lieve negli ultimi rollout). reward_events smoothed 0.0095 a fine run (era 0.0571 a 46% — il picco mid-training è dovuto al gym one-shot +400 esaurito, + Elm delivery da curriculum). in_battle smoothed 0.1845 (più alto di Agent 011 a causa del route_31 con encounter rate elevato). visited_tiles smoothed 176. Badge mai ottenuto.
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+62.0±81.0, avg tiles=224.7±85.8. Alta varianza (std±81) suggerisce policy non ancora stabile sulla navigazione dal punto di partenza.
- **Lesson**: tutti i waypoint del percorso sparano (Cherrygrove, Route 31, Violet City Main), ma il badge richiede battere Falkner — e il Pokemon lead è a livello 5. L'agente non ha informazione sul proprio livello nell'obs space, quindi non può valutare se è abbastanza forte per la palestra. Il route_31 bridge migliora il segnale di navigazione (reward_events peak più alto di Agent 011 in mid-training) ma non risolve il battle competence problem.

---

## Agent 013 (ex PPO_13) — Lead level in obs + heal reward + MAX_STEPS×4 (2026-05-27 → 2026-05-28)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- Curriculum: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 2×violet_city)
- **New obs feature**: `lead_level / 100.0` aggiunto all'obs vector (dim 11 → 12). RAM address: **0xDA49** (offset +0x1F da wPartyMon1=0xDA2A). Verificato empiricamente: start.state → lead_level=5 (Totodile lv5). Nota: indirizzo inizialmente impostato a 0xDA4B (byte Unknown nella struct) — corretto a 0xDA49 prima del training.
- **New reward**: heal reward +30 quando `hp_ratio - prev_hp_ratio > 0.4` fuori battaglia → incentiva uso Centro Pokémon prima della palestra.
- **MAX_STEPS**: 2^14 → 2^16 = 65,536 (4× più lungo). N_STEPS: 2048 → 4096 (doubled).
- **Result**: ep_rew_mean finale 62–66 (calo vs Agent 012 ~234 in TensorBoard). reward_events smoothed 0.0028 (quasi azzerato — waypoint e gym quasi mai raggiunti). value_loss stabile. Badge mai raggiunto.
- **Eval da start.state (10 episodi, MAX_STEPS=16,384 ridotto)**: badge=0/10, avg reward=+241.8, avg tiles=246.2.
- **Lesson**: il calo di ep_rew_mean è un artefatto di MAX_STEPS 4×: episodi più lunghi accumulano più step penalty (-0.01 × 65k = -655 max/ep vs -163 prima). L'eval con MAX_STEPS ridotto mostra +241.8 vs Agent 012 +62.0 — l'agente esplora di più. Il vero problema: step penalty -0.01 è eccessiva per episodi da 65k step e schiaccia il reward signal. Cambiata a -0.001 per Agent 014.

---

## Agent 014 (ex PPO_14) — Enemy obs + step penalty -0.001 (2026-05-28 → ...)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- Curriculum: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 2×violet_city)
- **Change 1**: step penalty -0.01 → **-0.001**. Penalità max per episodio: 65,536 × 0.001 = 65.5 (era 655).
- **Change 2**: obs dim 12 → **14**. Due nuove feature:
  - `enemy_lead_level / 100.0` — RAM `0xD0FC` (DataCrystal verified). Valore stale per ~100-500 step dopo BATTLE START (RAM non ancora inizializzata), poi stabile per tutta la battaglia. Verificato empiricamente: Falkner Pidgey→7, Pidgeot→9. ✓
  - `enemy_hp_ratio` — `(D0FF/D100) / (D101/D102)`. Scende man mano che si fa danno, torna a 1.0 quando esce il secondo Pokemon. Brevi drop a 0.0 durante animazioni/menu — normale. ✓
- Rationale: l'agente ora può stimare se è in vantaggio o svantaggio in battaglia (confrontando lead_level vs enemy_lead_level e i rispettivi hp_ratio). Segnale diretto per imparare "cura prima di entrare in palestra" e "attacca finché l'avversario ha hp alto".
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+299.6±61.3, avg tiles=304.0±58.0, avg steps=16384 (tutti troncati — eval usa MAX_STEPS=2**14=16384, diverso dal training 2**16=65536).
- **Training finale**: ep_rew_mean picco ~498 a 35M steps, poi exploitation collapse a ~386-395 a fine run. reward_events smoothed 0.0028 (identico a Agent 013 — nessun miglioramento). visited_tiles smoothed 243-282 (declino da 360 a metà training). ep_len_mean 34,500-36,000. entropy_loss stabile -2.01/-2.05. explained_variance 0.952-0.997.
- **Lesson**: exploitation collapse: l'agente esplora meno nel tempo — converge a "gironzola vicino alla partenza e sopravvive". Le feature nemico (enemy_lead_level, enemy_hp_ratio) non hanno sbloccato la navigazione perché l'agente non raggiunge il gym abbastanza spesso da usarle. Causa radice identificata: violet_city_gym.state peso 0 → il modello non ha mai allenato il combattimento diretto con Falkner in nessuno dei 14 training run.

---

## Agent 015 (ex PPO_15) — Damage reward + gym curriculum + N_STEPS 8192 (2026-05-29 → 2026-05-30, FERMATO a 58%)
- **500M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati), CHECKPOINT_FREQ=25M
- **Curriculum**: 2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + **1×violet_city** (era 2) + **1×violet_city_gym** (era 0, riattivato)
  - Nota sul gym env: il one-shot +400 non spara (mappa già in visited_maps all'init — Agent 008 lesson). Il per-episode +200 spara se l'agente esce e rientra. Beneficio principale: training diretto sui battle con i trainer in palestra e con Falkner → damage reward fire dall'env gym.
- **N_STEPS**: 4096 → **8192** (2**13). Con MAX_STEPS=65,536 e N_STEPS=8192, un episodio copre ~8 rollout (era ~16). PPO aggiorna i pesi vedendo più contesto per episodio → meno errori di bootstrap accumulati → migliore credit assignment verso reward lontani.
- **New reward**: damage reward. Formula: `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0` solo quando `battle_type > 0` per t e t-1. Guard `if delta > 0` per evitare false penalty su switch nemico. ✓ formula corretta.
- **Party levels in obs** (dim 15→20): 5 nuove feature agli indici 15-19 (slot 2-6 / 100.0). Struct size=0x30 verificato empiricamente (start.state → [5,0,0,0,0,0] ✓).
- **Gym trainer flags** (empiricamente verificati via `test_enemy_level.py`):
  - Trainer 1 beaten: `0xD836 bit=4` (mask 0x10, flag #1020) → +100 reward (RISE)
  - Trainer 2 beaten: `0xD836 bit=3` (mask 0x08, flag #1019) → +100 reward (RISE)
  - Falkner beaten: `0xD84E bit=5` (mask 0x20, flag #1213) — coperto dal badge reward, non reward separato
- **Gym battle exit reward** +150: map-constrained a (10,7). Max 3×/episodio.
- **Stuck penalty** -0.02/step su ogni tile rivisitata nell'episodio.
- **Result**: FERMATO a 289M/500M step (58%). Progressione ep_rew_mean: +30 a 13% → −18 a 26% → stagnante a −170/−183 da ~130M step in poi. policy_gradient_loss sceso a −0.001 (quasi zero). explained_variance stabile 0.99+ (value overfit). visited_tiles cresciuti da 163 (26%) a 250 (58%) — l'agente esplorava di più, ma veniva penalizzato di più. Badge mai raggiunto.
- **Root cause**: stuck_penalty miscalibrata. Conto per episodio medio (29k step, 260 tile nuove): new tiles +260, stuck −575, step −29, events +170 → totale −174. Il penalty da solo pesa 575 vs 260 di exploration reward (ratio 2.2:1 a sfavore). Breakeven: stuck_penalty < 260/28,740 ≈ 0.009/step. Con -0.02 non c'è mai un incentivo netto a esplorare nuovi territori.
- **Lesson**: stuck_penalty deve essere abbastanza piccolo da lasciare reward netto positivo in un episodio di pura esplorazione. -0.02 è 7× sopra il breakeven. La conseguenza: il training converge su "vai presto e muori" (episodi corti) o "gira sul noto e accumula penalty". Il valore corretto è ~−0.003, che porta il baseline a +315/ep e mantiene la pressione verso nuove aree senza annullare il signal degli eventi.

---

## Agent 016 (ex PPO_16) — Stuck penalty calibrata (2026-05-30 → 2026-06-02, COMPLETATO)
- **500M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192 (invariati)
- **Curriculum**: invariato da Agent 015
- **Unica modifica**: `stuck_penalty -0.02 → -0.003` (7× ridotto)
- **Training final**: ep_rew_mean **+523** (vs −180 di Agent 015 — calibrazione confermata efficace). visited_tiles smoothed 244-319. policy_gradient_loss −0.0001/−0.001 (policy quasi convergente). explained_variance 0.98+ (value function molto fitted). in_battle smoothed **0.124** (12.4% del tempo in combattimento — sintomo di local optima wild battle). hp_ratio 0.77-0.91. reward_events 0.0046-0.0147 (curriculum-driven).
- **Eval da start.state (10 episodi)**: badge=**0/10**, avg reward=**+180.5±92.0**, avg steps=20,499±17,054, avg tiles=261±82.5. Episodi 4 e 6 raggiungono reward 317-349 ma nessun badge.
- **Gap training/eval**: +523 (training) vs +180 (eval start.state). Le curriculum states `violet_city_gym.state` sparano eventi gratis che gonfiano ep_rew_mean ma la policy non transfer al percorso completo.
- **Failure mode osservato** (gameplay manuale post-training): l'agente entra nell'erba, ingaggia wild Pokémon, perde la battaglia → HP=0 → episodio termina prima del gym. Il `damage_reward` (k=5.0) introdotto in Agent 015 ha creato un attrattore locale: combattere wild = reward immediato, navigare = reward distante.
- **Root cause**: il reward locale del damage in wild battle (+5.0 × delta_hp per turno) compete con il reward distante del gym. PPO greedy → l'agente preferisce l'erba.
- **Lesson**: il damage_reward non distingue tra battaglie utili (gym) e dannose (wild). Va rimosso, e le wild battles fuori dal gym vanno penalizzate esplicitamente per spezzare il local optima.

---

## Agent 017 (ex PPO_17) — Wild battle penalty -3.0 (2026-06-02, FERMATO a 22% / 44.6M step)
- **200M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192
- **Curriculum**: invariato da Agent 016
- **Modifica 1**: Wild battle penalty `−3.0/step` quando `battle_type == 1 AND (map_bank, map_number) != (10,7)`
- **Modifica 2**: Damage reward RIMOSSO completamente
- **Result a 44.6M step (22%)**: FERMATO per stallo oscillante. ep_rew_mean oscilla nel range **[-5000, -2000]** senza trend monotonico (best -1973 a 27M, regressione a -3705 a 44M). `in_battle` oscilla 0.003↔0.083, `visited_tiles` 94↔288, `reward_events` quasi sempre 0.
- **Root cause (calcolo)**: il penalty −3.0/step è strutturalmente troppo aggressivo. Una wild battle media (~80 step pyboy) = −240. Per attraversare Route 30+31 sono fisiologiche 5-10 wild battle inevitabili → costo −1200/−2400 vs +455 totali di waypoint reward dal start a Violet Gym. L'agente fa la matematica corretta: muoversi costa più di stare fermo → equilibrio di stallo.
- **Signal pattern identificato**: "stallo oscillante" ≠ "stagnazione" ≠ "collapse". entropy_loss -2.04 sano, value_loss basso, MA reward oscilla senza convergere. Firma diagnostica: penalty calibrato male rispetto al reward landscape.
- **Lesson**: il wild penalty deve essere abbastanza forte da disincentivare grinding ma abbastanza piccolo da non sopprimere la navigazione. -3.0 è 3× sopra il break-even. Calibrato a -1.0 in Agent 018.

---

## Agent 018 (ex PPO_18) — Wild battle penalty calibrata a -1.0 (2026-06-03, FERMATO a 51% / 103M step)
- **200M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192 (invariati da Agent 017)
- **Curriculum**: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 1×violet_city + 1×violet_city_gym)
- **Unica modifica vs Agent 017**: `wild_battle_penalty −3.0 → −1.0` (3× ridotto)
- **Result a 103M step (51%)** — dati last-100 rollouts:
  - `ep_rew_mean` mean=**−1160**, std=166 (range stabile [-1326, -994])
  - `in_battle` mean=**0.0495** (2.5× meglio di Agent 016 a 0.124) ← fix wild penalty funziona
  - `visited_tiles` mean=**214** ± 43 (esplorazione limitata ma stabile)
  - `reward_events` mean=**0.0055** ± 0.009 (sparsi waypoint, non consistenti)
  - `hp_ratio` mean=**0.90** ± 0.06 (sopravvivenza ottima)
  - Best ep_rew_mean: -587 a 49M (mai più raggiunto)
- **Confronto Agent 017 vs Agent 018 a parità di step**:
  - 27M: Agent 017 -1973, Agent 018 -1348 (+32%)
  - 44M: Agent 017 -4184, Agent 018 -1501 (+64%)
  - 50M: Agent 017 -3500, Agent 018 -956 (+73%)
  - Math della calibrazione era corretta, ma non sufficiente.
- **Root cause: "stable suboptimal convergence"**. La policy ha convergato a un equilibrio negativo. Diagnostica:
  - `explained_variance` 0.99+ → value function ha overfittato ai return correnti
  - `policy_gradient_loss` -0.001 → gradiente di policy quasi morto
  - `entropy_loss` -2.04 stabile → la policy non sta più esplorando attivamente
  - `ep_rew_mean` std/mean = 14% → bassa varianza, comportamento ripetitivo
  - Pattern definitivo: non è oscillazione (Agent 017), non è collapse (Agent 002), non è grinding (Agent 009). È **convergence to local optimum**.
- **Lesson finale del filone MLP**: dopo 18 run con tutte le combinazioni di reward shaping (curriculum, milestones, calibration), observation features (lead_level, enemy_obs, party_levels, flag bits), e hyperparameter tuning (gamma, ent_coef, n_steps), il badge da start.state non è mai stato ottenuto in evaluation. Il limite **non è di reward engineering**: lo state vector `(map_id, x, y, hp, levels, ...)` non porta informazione spaziale sufficiente perché PPO costruisca un piano di navigazione di 800+ step verso il gym. CnnPolicy vede il sentiero, l'erba, gli NPC e la porta del gym — informazione che il vettore non contiene.
- **Decisione 2026-06-03**: chiusa la fase MlpPolicy. Switch a **CnnPolicy + frame stacking** (Agent 019+, vedere sezione dedicata).

---

## Agent 019 (ex PPO_CNN_1) — Primo run CNN, stable suboptimal (2026-06-03, FERMATO a 4M / 8%)
- **50M steps**, 4 envs, gamma=0.999, gae_lambda=0.95
- **Curriculum**: 4×start.state (puro, no curriculum diversity)
- **Hyperparameters**: lr=2.5e-4, n_steps=2048, batch=256, n_epochs=4, ent_coef=0.01
- **Frame stack**: 4, obs (72,80,3) uint8 → (12,72,80) dopo transpose
- **Result a 4M (8%)**: FERMATO per stable suboptimal. ep_rew_mean=-502.6 ± 5.7 (last-100). Convergenza precoce a -500.
  - `explained_variance` -1.15 oscillante (era 0.88 inizio)
  - `clip_fraction` 0.36 (sopra clip_range 0.2)
  - `approx_kl` 0.08 (5× soglia raccomandata)
  - `reward_events` mai > 0 (mai raggiunto waypoint)
- **Lesson**: senza curriculum diversity, anche la CNN converge a "stay at New Bark Town" — il problema NON era solo la rappresentazione, era anche l'assenza di segnale di reward "vicino". Confermato che curriculum è essenziale per credit assignment a lungo orizzonte.

---

## Agent 020 (ex PPO_CNN_2) — Curriculum diverso + 8 envs + clip stretto (2026-06-03 → 2026-06-04, CRASH a 95% ma SUCCESS) ⚡

**Primo run in 20 (Agent 001..18 MLP + Agent 019) a raggiungere ep_rew_mean POSITIVO da start.state curriculum.**

- **50M steps**, **8 envs**, gamma=0.999, gae_lambda=0.95
- **Curriculum** (vs Agent 019):
  - 3× start.state (main target, 37.5%)
  - 1× mid_route30 + 1× route_31 + 1× before_elm_delivery + 1× violet_city + 1× violet_city_gym
- **Hyperparameter fix vs Agent 019**:
  - `LEARNING_RATE_CNN` 2.5e-4 → **1.5e-4** (più conservativo)
  - `BATCH_SIZE_CNN` 256 → **512** (sfrutta GPU)
  - `clip_range` 0.2 → **0.1** (update meno aggressivi)
- **Result a 47.4M (95%, CRASH per bug GIF path)**:
  - `ep_rew_mean` BEST: **+112.68** at step 41.5M
  - `ep_rew_mean` Last-100: **+53.6 ± 18.9** (PRIMO POSITIVO STABILE in 20 run)
  - `visited_tiles` Last-100: 296 ± 47 (vs Agent 018 a 214) — esplora 40% più ampio
  - `in_battle` Last-100: 0.045 (vs Agent 018 a 0.05) — evita wild battle
  - `reward_events` Last-100: 0.002 (raramente ma occasionalmente raggiunge waypoint)
  - `explained_variance` Last-100: 0.57 ± 0.22 (riprese stabilità verso fine)
  - `entropy_loss` Last-100: -1.53 ± 0.03 (policy specializzata ma non collassata)
  - `fps` 2,500-2,600 (8 envs × CnnPolicy + RTX 5080)
  - Tempo totale: ~5h per 47M step
- **Trend ep_rew_mean (storia completa)**:
  ```
  start: -3,595 → 4M: -676 → 11M: -129 → 15M: +10 (primo positivo) →
  18M: +26 (picco intermedio) → 22M: +23 → 27M: -166 (regressione) →
  34M: -94 → 39M: +20 → 43M: +68 → 47M: +36 → Last-100: +53.6
  ```
- **Crash analysis**: `FileNotFoundError: '/home/fabio/Projects/runs/gifs'` al primo save GIF (env 0 al 100° episodio, ~47.4M step). Bug del default `gif_dir="../runs/gifs/"` (relativo doppio-punto). Fix: default → `None`.
- **Checkpoint salvati**: `runs/checkpoints/agent_020/agent_020_20000000_steps.zip` (al picco intermedio +26) e `agent_020_40000000_steps.zip` (vicino al picco massimo +68).
- **Lesson**:
  1. Il **curriculum** + **clip range stretto** + **batch più grande** = combo vincente. Il PPO ha imparato per la prima volta a navigare in modo significativo.
  2. `clip_fraction` resta alto (0.34) anche con clip_range=0.1, ma il training rimane stabile — sintomo che la policy continua a evolversi attivamente, non collasso.
  3. `reward_events` sempre basso (0.002) MA `ep_rew_mean` positivo significa che il guadagno viene da **exploration tiles** + **eventi rari** + risparmio penalty. La policy NON ha ancora "scoperto" Violet City Gym da start.state.
- **Next step**: evaluation da `start.state` su 10 episodi del checkpoint 40M. Se badge > 0 → success. Se badge = 0 → Agent 021 con focus su credit assignment del segmento finale (start → gym).

---

## Agent 021 (ex PPO_CNN_3) — Damage reward al gym + heal reward (2026-06-04, COMPLETATO)

**Eval da start.state: badge=0/10. L'agente impara a esplorare ma non combatte mai (in_battle=0%).**

- **50M steps**, 8 envs, lr=1.5e-4, n_steps=2048, batch=512, n_epochs=4, ent_coef=0.03, clip_range=0.1
- **Curriculum**: invariato da Agent 020 (3×start + 1×mid_route30 + 1×route_31 + 1×before_elm_delivery + 1×violet_city + 1×violet_city_gym)
- **Modifica vs Agent 020**: damage reward RE-INTRODOTTO ma **map-constrained al gym** (10,7): `(prev_enemy_hp_ratio - enemy_hp_ratio) * 10.0` solo se `battle_type > 0` per t e t-1 E mappa == (10,7). Evita il local optima wild di Agent 016 (lì il damage era globale).
- **Result training**: ep_rew_mean picco ~+250 a ~32M step, poi declino lento a fine run. visited_tiles smoothed ~466 (esplorazione molto ampia — record CNN). `in_battle` smoothed ~0.0 → l'agente evita sistematicamente ogni combattimento.
- **Eval da start.state (10 episodi, 3 checkpoint)**: badge=**0/10** su tutti. avg reward ~+175/+188, **in_battle = 0% in OGNI episodio**.
- **Gameplay osservato (--watch)**: l'agente NON si muove da New Bark Town. Esplora le case, rientra in casa propria, sale/scende i piani, ma **non entra mai nell'erba** (obbligatoria per raggiungere Cherrygrove). Non è un bug di in_battle=0: è una policy che massimizza i tile reward dentro New Bark senza mai partire.
- **Root cause**: il reward landscape penalizza l'uscita. New Bark Town + interni offrono ~+250 di tile reward "gratis" e sicuri; uscire verso Route 29 → wild battle penalty (-0.05/step) + nessun tile nuovo immediato che batta i +250 già accumulati. Il damage reward al gym è irraggiungibile da start.state: l'agente non arriva mai al gym, quindi quel segnale non propaga.
- **Lesson**: il damage reward map-constrained al gym è inerte da start.state (credit assignment troppo lungo). Il vero problema è che la navigazione iniziale (uscire da New Bark) non è mai incentivata abbastanza forte da superare l'attrattore locale dell'esplorazione interna. Serve un **mega-bonus per waypoint** che renda raggiungere Cherrygrove/Route 31/Violet enormemente più redditizio del gironzolare. → Agent 022.

---

## Agent 022 (ex PPO_CNN_4) — Mega-bonus waypoint per-episodio (2026-06-04 → 2026-06-05, COMPLETATO 100M)

**Eval da start.state: badge=0/10. L'agente ORA combatte (in_battle 3.2%) ma il comportamento è "schizofrenico" tra due policy in conflitto — peggiore in eval dei CNN precedenti.**

- **100M steps** (override manuale, config dice 100M), 8 envs, hyperparameters invariati da Agent 021
- **Curriculum**: invariato
- **Modifica vs Agent 021**: per-episode waypoints potenziati **~80×** per spezzare l'attrattore "resta a New Bark":
  - Cherrygrove (26,3): +25 → **+500**
  - Route 31 (26,1): +50 → **+300**
  - Violet City West (26,2): +80 → **+400**
  - Violet City Main (10,5): +100 → **+500**
  - Gym (10,7): +200 (invariato)
- Wild penalty ridotto progressivamente in questo filone a **-0.05/step** (da -1.0 MLP) per non sopprimere la navigazione.
- **Result training**: ep_rew_mean picco assoluto +994 a 507k (curriculum), peak reale stabile **+739 a ~55M**, poi **declino a +509 a 72M** (-230 vs peak). Last-100 a fine run ~+626 ± 41. `hp_ratio` Last-100 ~0.91, `in_battle` Last-100 ~0.058 (finalmente non-zero — l'agente combatte), `reward_events` ~0.008, `explained_variance` 0.25 ± 0.64 (INSTABILE, vs 0.99 dei run convergenti).
- **Eval da start.state**:
  - checkpoint 40M: avg reward **+81 ± 86**, in_battle 3.1%, badge **0/10**, avg tiles 406
  - checkpoint 60M: avg reward **+89.6 ± 53**, in_battle 3.2%, badge **0/10**, avg tiles 445
  - Range per episodio: +1.3 → +180.2. Nessun episodio raggiunge il gym.
- **Confronto con Agent 020/Agent 021**: Agent 022 in eval è **PEGGIORE** (+89 vs +188/+196) nonostante il training salga a +626. La std scende da 86 (40M) a 53 (60M): la policy si consolida su un equilibrio mediocre "combatti un po', esplora moderato, mai vince", abbandonando l'exploration policy stabile e ad alto reward di Agent 021.
- **Root cause: policy segregation visiva (limite strutturale, NON di reward shaping)**. La CnnPolicy impara policy separate per aspetto visivo: gli env curriculum (che partono già avanti) imparano a combattere e raccolgono i mega-bonus → ep_rew_mean training sale; gli env start.state imparano a esplorare. Le due policy **non transferiscono** perché la rete le associa a scene visive diverse. Il mega-bonus 80× ha cambiato il comportamento nei curriculum env (in_battle ora >0) ma ha destabilizzato la policy start.state (explained_variance crolla a 0.25). Pattern di declino post-peak identico a Agent 021.
- **Lesson finale del filone reward-shaping CNN**: dopo Agent 020/3/4, tre run consecutivi 0/10 badge da start.state con lo stesso pattern strutturale (training sale, eval non riflette, policy segregation). Il reward shaping ha raggiunto il suo limite: aumentare i waypoint cambia COSA fanno i curriculum env, non fa GENERALIZZARE la policy a start.state. Il segnale di reward denso non basta — manca un segnale di **novelty intrinseco e visivo** che spinga la policy start.state a uscire e continuare a esplorare territorio nuovo a prescindere dai waypoint hardcoded.
- **Decisione 2026-06-05**: pivot a **KNN visual novelty** (approccio Whidden / PokemonRedExperiments). → Agent 023.
  - **CORRECTION 2026-06-09**: after inspecting the reference repo, Whidden's **V2** (the version that
    reaches Cerulean = past the 1st gym) **replaced KNN with coordinate-based exploration** — which this
    project already has (`visited_tiles` keyed on `(map_bank, map_number, local_x, local_y)`). KNN is the
    *abandoned* V1. So Agent 023 does **not** add KNN; it realigns the existing setup to the V2 recipe.

---

## Agent 023 (ex PPO_CNN_5) — Whidden V2 realignment: reward rescale + pure single-start (2026-06-09, RECIPE LOGGED, RUN PENDING)

**Hypothesis**: the blocker across Agent 020..4 was not representation (env is already CNN + frame-stack like
PokemonRedExperiments) nor "reward-shaping exhaustion", but five concrete divergences from Whidden V2:
inverted reward scale, heterogeneous curriculum (policy segregation), no level/opponent reward, no
self-state in obs, over-long episodes. Agent 023 fixes scale + curriculum + level reward + episode length
first (cheap validation); self-state in obs and battle competence are deferred to Phase 4.

- **30M steps (SHORT VALIDATION)**, **12 envs**, lr=1.5e-4, n_steps=2048, batch=512, n_epochs=4,
  ent_coef=0.03, clip_range=0.1, gamma=0.999, gae_lambda=0.95, VecFrameStack(4). CHECKPOINT_FREQ=5M.
- **Curriculum**: PURE single start — **12× start.state** (was 3×start + 5 mixed). Removes the
  visually-keyed sub-policies that never transferred to start.state in eval (the Agent 022 segregation cause).
- **Reward realigned (`env/rewards.py`)** — every term single-digit, global `REWARD_SCALE = 0.1`:
  - new tile **+0.02** (dense PRIMARY driver) · new map first-entry (lifetime) **+1.0**
  - level-sum increase **+1.0 × Δmax** · opponent level **+0.2 × Δmax** (in-battle only, clamped 1..100)
  - story event count **+1.0 × Δmax** (latched flags: egg/elm/trainer1/trainer2) · rival **+1.0** (latched falling edge)
  - heal out of battle **+2.0** · death **−1.0** · badge (zephyr) **+10.0**
  - **REMOVED**: per-episode mega-waypoints (+300/+500), wild-battle penalty, flat step penalty, catch
    bonus, big one-shot map milestones. **DEFERRED to Phase 4**: gym damage reward, gym battle-exit reward.
  - Running maxima tracked via new `reward_maxes` dict (`make_reward_maxes`, reset per episode), mutated
    in place by `compute_reward` — same pattern as `visited_maps`/`episode_maps`. Optional arg (defaults
    None) so the dead MLP env still imports.
- **`norm_reward=False`** (`train_cnn.py`): rewards are now well-scaled, so reward normalization is
  unnecessary — and was harmful before (running-std dominated by +1000 spikes crushed the dense signal).
- **MAX_STEPS 2¹⁶ → 2¹⁵** (32768): more episode resets → better credit assignment.
- **Bug fixes (Phase 0)**:
  - GIF crash: `train_cnn.make_env` now passes `gif_dir=None` (the relative `../runs/gifs` default crashed
    Agent 020 at 95%).
  - eval badge detection: `evaluate_cnn` now reads `infos[0]["zephyr"]` (terminal-step info), not a
    post-auto-reset RAM read — the old code returned `badge=no` even on a win.
  - op_level stale-RAM bug: enemy level (`0xD0FC`) is garbage outside battle; reward now gated on
    `battle_type > 0` + legal-range clamp.
- **New instrumentation**: per-EPISODE navigation metric `nav/reach_{cherrygrove,route31,violet_west,
  violet_main,gym}` (fraction of episodes reaching each waypoint) + `nav/ep_max_waypoint`. Logged on
  episode end (NOT step-averaged), so it is un-confounded by dwell time. This is the honest success signal
  — `ep_rew_mean` misled Agent 020..4 (training rose while eval stayed 0/10).
- **Success gate**: `nav/reach_violet_main` trending → ~0.5 by 30M ⇒ recipe works ⇒ scale up (Phase 5) +
  Phase 4 battle competence. Flatline at 0 ⇒ revisit the tile weight before spending more compute.
- **Result (COMPLETED 30M, 2026-06-09)**: realignment WORKED at the hard part, but the frontier STALLED at Route 31.
  - `nav/reach_cherrygrove` → **1.0** (100% of episodes from ~6.6M); `nav/reach_route31` firmed to **1.0** by 30M.
  - `nav/reach_violet_west` / `reach_gym` = **0.000 through the final rollout**; `nav/ep_max_waypoint` capped at exactly **2.0** (zero variance).
  - `ep_rew_mean` plateaued ~1.75 (flat the entire second half); `entropy_loss` -1.89 (healthy, NO collapse);
    `visited_tiles` ~600-850; `in_battle` ~0.15 (fights wild battles without dying, hp_ratio ~0.9).
  - **First run in 23 to escape New Bark and reliably hold 40% of the route** — no collapse, no policy segregation.
- **Root cause of the stall**: exploration novelty was **EPISODE-scoped** → re-treading the known New Bark→Route 31
  corridor pays the full tile reward every episode → no directional gradient to push into unexplored Violet. The
  lifetime map bonuses that drove segment 1 are consumed. (Whidden V2 uses **persistent** coordinate novelty to
  avoid exactly this — only genuinely-new territory pays, so the frontier keeps advancing.)
- **Bug found**: `CHECKPOINT_FREQ_CNN` is counted by SB3 in callback-calls (= timesteps / n_envs), so 5M × 12 envs
  = 60M > 30M total → **no intermediate checkpoints saved, only `agent_023_final.zip`**.
- **Lesson**: scale realignment + single-start solved escape + segment-1 navigation, but a flat per-tile exploration
  reward cannot cross distant chokepoints. → **Agent 024**: hybrid exploration (small episode-new +0.005 to keep the
  corridor warm + dominant **lifetime-new +0.02** for frontier expansion), **warm-started from `agent_023_final`**,
  with the checkpoint-frequency bug fixed (save_freq divided by N_ENVS).

---

## Agent 024 (ex PPO_CNN_6) — Lifetime tile-novelty (warm-start from Agent 023), STOPPED ~55% (2026-06-09)

**Pure single-start + lifetime novelty → reward death-spiral; frontier still walled at Route 31.**

- 30M target (stopped ~16.7M), 12× start.state, warm-started from `agent_023_final`.
  Reward: episode-new **+0.005** + lifetime-new **+0.02** (×0.1).
- **Result**: `nav/reach_violet_west` = **0.000** throughout; `ep_max_waypoint` capped at 2.0 (same wall as Agent 023).
  `ep_rew_mean` **declined monotonically +2.59 → +0.38**; `visited_tiles` 570 → 380; `ep_len_mean` 32768 → 28565.
- **Root cause**: lifetime novelty SATURATES. Once the known corridor is "seen forever", re-tread pays only the
  tiny +0.005 trail reward and the lifetime bonus never fires at the *undiscovered* frontier — so the dense corridor
  reward that kept Agent 023 busy was stripped away with nothing to replace it. The agent contracted into a smaller,
  lower-reward routine. Novelty only pushes a frontier the agent OCCASIONALLY crosses; it can't CREATE discovery.
- **Bonus**: confirmed the checkpoint-freq fix (intermediate checkpoints at 3/6/9/12/15M saved).
- **Lesson**: reward-shaping (episode vs lifetime novelty) cannot solve a DISCOVERY problem. Reverted in Agent 025.

---

## Agent 025 (ex PPO_CNN_7) — Small curriculum + realigned reward (warm-start), COMPLETED 30M (2026-06-09 → 2026-06-10)

**Curriculum made the start-state policy CATASTROPHICALLY FORGET segment-1 navigation — exposed only by the new start-state-filtered nav metric. `ep_rew_mean` ~1.5 hid the regression completely.**

- 30M, 12 envs = **8× start + 2× route_31 (=Violet West 26,2) + 2× violet_city (10,5)**, warm-started from `agent_023_final`.
- Reward reverted to Agent 023 episode-novelty (**+0.02/new tile**, ×0.1). `nav/reach_*` filtered to start.state episodes
  (`info["from_start"]`) so curriculum envs (which begin past the waypoints) can't inflate it.
- **Result**: `ep_rew_mean` stayed a healthy ~1.5 (LOOKED fine), but the start-state nav metric **COLLAPSED**:
  `nav/ep_max_waypoint` 2.0 (warm-start) → **0.0 by ~4M**, flat for the rest of the run; `nav/reach_cherrygrove`
  and `reach_route31` fell **1.0 → 0.0**. `reach_violet_west`+ = 0 throughout. The start policy forgot how to leave New Bark.
- **Root cause**: curriculum envs start in UNEXPLORED Violet (high novelty reward) while start envs re-tread KNOWN
  New Bark→Route 31 (low reward). This reward asymmetry makes PPO's shared-policy updates favor Violet behavior,
  degrading start-state navigation. The realigned reward reduced but did NOT eliminate the asymmetry — fresh
  territory always out-earns known territory. `ep_rew_mean` masked it because the curriculum envs kept earning;
  the `from_start` filter is the only reason the regression was visible.
- **Lesson**: even a SMALL fixed curriculum on the realigned reward sacrifices the start-state policy (segregation /
  catastrophic forgetting). **Fixed curricula are ruled out for the discovery problem.** Three approaches now failed on
  the same structural tension: Agent 023 (stall — can't discover), Agent 024 (death-spiral — novelty saturates),
  Agent 025 (forgetting — curriculum segregates). → architectural pivot needed: **frontier state-sharing**
  (Go-Explore-lite), where reset states are the start policy's OWN trajectory edge — continuous with start-state
  experience, so no high-reward island and no segregation.

---

## Agents 026–028 (ex PPO_CNN_8 stage1/stage1b/crossing) — Egg-quest STORY GATE + reverse curriculum (2026-06-09 → 2026-06-10, IN PROGRESS)

**THE big discovery (user's gameplay knowledge): the "Route 31 wall" of Agent 023/6/7 is a STORY GATE, not a
navigation chokepoint.** Two trainers block Route 30's WEST branch until the Mystery Egg is delivered to Elm
(confirmed: Bulbapedia + gameplay). The agent never did the egg quest (`reward_events`=0) → permanently
blocked. This OVERTURNS the Agent 023/6/7 "discovery" diagnosis. (The Agent 025 "frontier state-sharing" plan was
abandoned the moment the gate was found — state-sharing can't open a scripted gate.)

**Map constants RE-VERIFIED 2026-06-10 by walking the route (instrumented save_state.py). Old labels WRONG:**
(26,3)=Cherrygrove · (26,1)=Route 30 north / GATE zone · (26,2)=Route 31 POST-GATE (Dark Cave area) ·
(26,11)=Route31↔Violet gatehouse · (10,5)=Violet · (10,7)=Gym · (26,10)=Mr.Pokemon's house · (3,70)=Dark Cave.
Old code had `ROUTE_31`=(26,1) and `VIOLET_CITY_WEST`=(26,2) — both wrong; fixed. Route 30 forks: WEST→Route 31
(gated), EAST→Mr.Pokemon's house (dead-end).

**Approach: REVERSE CURRICULUM** — single start-state per run (→ no segregation), warm-start chain. Egg events
weighted up: `egg_received`+3, `egg_delivered`+5 (opens gate); Mr.Pokemon house +2 gated on "egg not yet received".

- **Stage 1** (`egg_delivered`, warm Agent 023): reached (26,1) gate zone but did NOT cross to (26,2) even with the
  gate OPEN. Causes: (a) a BUG — Mr.Pokemon house +2 fired post-egg, luring the agent to the EAST dead-end off
  the correct WEST path; (b) warm-start aversion. `ep_max_waypoint`=2 flat.
- **Stage 1b** (Mr.Pokemon reward gated on egg-not-received): agent now TRIED the west path — but DIED in the
  battle gauntlet. `ep_len_mean` crashed 32768→17500 (early termination = blackouts), `in_battle` 0.5, `hp` 0.6.
  `reach_route31` max 0.33, never consolidated. → **BATTLE COMPETENCE is a wall on the ROUTE itself**, not just at the gym.
- **Crossing stage** (`crossing.state` = (26,1) doorstep PAST the west trainers, Totodile lv11): reliably crosses
  to Route 31 (`reach_route31`=1.0), SURVIVES (`ep_len`=32768, `in_battle`~0.05, `hp`~0.98), explores Route 31
  (~700 tiles) — but stalls at Route 31 (26,2) → Violet (10,5): `reach_violet`=0 flat. New chokepoint = the
  GATEHOUSE building door (26,11).

**Pattern / lessons (Agent 026):**
- The agent broadly navigates but stalls at EVERY map-to-map transition, especially BUILDING DOORS (gatehouse,
  exiting Mr.Pokemon's house). The flat per-tile exploration reward gives no gradient toward the specific exit/door tile.
- BATTLE COMPETENCE (winning route trainers / surviving the grass) is a real wall, hit on the route before the
  gym. Needs HP/level in obs + a battle reward (deferred Phase-4 work, now critical).
- The reverse curriculum advances the frontier one chunk per run (west gauntlet → Route 31 → now Route31→Violet),
  monotonic but slow (~1 chokepoint per 3h run).
- **Next idea**: modest per-episode LATCHED waypoint rewards (Route31/gatehouse/Violet/gym, +2 each) — single-start-safe
  (no segregation), latched (no cycling), small (no scale domination; the log even notes "per-episode is ok if small")
  — to create a directional gradient through the map-door chokepoints.

---

## Agent 029 (ex PPO_CNN_8_crossing_wp) — Waypoint rewards from crossing.state (2026-06-10 21:36 → 06-11 00:05, STOPPED 27.4M/30M)

Warm variant of the crossing stage WITH the new per-episode latched waypoint rewards (+2 each).

- **Waypoint rewards WORK as a door gradient**: `reach_route31`=1.0 stable, `ep_max_waypoint`=3,
  `ep_rew_mean` 1.73 (peak 2.16), full survival (`ep_len`=32768, `hp_ratio`=1.0).
- **But**: `reach_violet`=0 flat for the whole run — the Route31→Violet GATEHOUSE door (26,11) was never
  crossed despite its +2 waypoint waiting. The +2 (0.2 post-scale) is apparently too small vs ~700 tiles
  (1.4 post-scale) of comfortable Route 31 exploration income.
- **in_battle → 0.004**: TOTAL battle avoidance learned. No reward pays for fighting; battles only cost HP.

## Agent 030 (ex PPO_CNN_9_selfstate) — First Dict-obs run, cold start (2026-06-11 00:08 → 02:22, STOPPED 23.8M/30M)

First run with the new Dict observation (image + 7-float self-state vector) + MultiInputPolicy, cold start
(old CnnPolicy checkpoints incompatible). Log: `runs/cnn9_train.log`.

- Nav metrics PEAKED mid-run (`reach_route31` max 1.0, `ep_max_waypoint` max 3) then **collapsed to 0** by
  the end. `ep_len_mean` fell 32768 → 22.7k (early terminations = blackouts), `in_battle` 0.097, final
  `ep_rew_mean` 1.32 (peak 1.70).
- **Pattern: battle-incompetence death spiral** — the agent pushes the frontier, gets killed by trainers/wilds,
  the policy regresses to safe wandering. Same wall as stage1b, now visible end-to-end in one run.

## Agent 031 (ex PPO_CNN_9_gymtest) — Phase-A battle test INSIDE the gym (2026-06-11 02:44 → 03:37, STOPPED 10.3M/30M) ⚠️ KEYSTONE

Cold start from `violet_city_gym.state` (2 steps inside Falkner's gym): pure battle-competence test, no grass.

- **The agent WALKS OUT of the gym and tours Violet City**: visited_tiles 614–866 (the gym alone has ~100),
  `in_battle` peaked 0.63 early then decayed to 0.11, `ep_len` ~31.7k ≈ always truncation, **badge never won**,
  `ep_rew_mean` 2.80 (earned by sightseeing, not fighting).
- **Direct reward arithmetic confirms it is RATIONAL**: touring Violet ≈ 600 tiles × 0.02 = 12 pre-scale,
  vs full gym fight chain = gym damage 6 + badge 10 = 16 pre-scale but discounted by risk of death and battle
  length — exploration income structurally dominates combat income. The same inequality explains ALL the
  avoidance/grinding failure modes of the last 30 runs.
- **Conclusion**: this is not a representation or curriculum problem anymore — it is a reward-geometry problem.
  → full redesign (Agent 033) instead of further mini-patches.

---

## Agent 033 (ex PPO_CNN_10) — REDESIGN: event-dominant reward + story/combat obs + 150M single run (2026-06-11, PLANNED)

Step-back redesign aligned with what actually worked in PokemonRedExperiments (PWhiddy) where events/badges
dominate exploration ~10–50× cumulatively. One coherent change-set, then ONE long run with abort gates,
instead of more 30M patch-runs. Plan file: `~/.claude/plans/rispetto-a-quello-che-sleepy-brook.md`.

**Reward (pre-scale, REWARD_SCALE 0.1 unchanged)**: tiles +0.02 / new map +1 / waypoints +2 / Mr.Pokemon +2 /
heal +2 / death −1 / op-level +0.2Δ / gym damage 3×Δ all KEPT. Raised: egg received 3→**8**, egg delivered
5→**12**, rival 1→**3**, gym trainers 1→**5** each, badge 10→**30**. NEW: Route 30/31 trainer beaten +**5**
each (×4: Joey, Mikey, Don, Wade — flags to verify via pret/pokegold + savestate diff + manual ram_scan
session). Level cap (lead≤13 hack) → **saturation** `scaled = min(s, 15 + (s−15)/4)`, reward +1×Δscaled.

**Sanity math (per-episode, pre-scale)**: wander ≈ 20 · grind ≈ 19 · avoid-battles ≈ 33 · **story-optimal ≈ 136**
(dominates 4–7× globally; locally at the gym: fight chain 55 vs tour-Violet 12).

**Obs**: vector 7→11 (+egg_received, +egg_delivered, +route_trainers/4, +gym_trainers/2 — the Route 30 fork is
visually identical pre/post delivery, the policy literally cannot condition on it from pixels). Image
(72,80,3)→(72,80,4): **visited-mask channel** (PWhiddy mechanism) for an explicit frontier gradient at doors.
MAX_STEPS 2^15→**2^16** (optimal path ≈14.6k steps; old limit left 2.2× slack for backtrack+battles).

**Run**: 12×start.state only, hyperparams UNCHANGED (attribution), 150M budget (~15h @ ~2700fps), ckpt 5M,
go/no-go gates: 10M reach_cherrygrove≥0.8 & egg_received>0 · 25M egg_delivered≥0.3 · 50M reach_violet≥0.3 &
in_battle∈[0.05,0.3] · 100M badge_rate>0 (else if reach_gym>0.3: bump gym damage 3→5, badge 30→50, warm-restart).
New telemetry: nav/egg_received_rate, egg_delivered_rate, route_trainers_mean, gym_trainers_mean, badge_rate.

**Ruled out by this redesign analysis**: resuming Agent 026 with higher ent_coef (east-fork bias is in
the weights; stage1b already was that experiment post-bugfix and hit the battle wall) · catch/evolution rewards
(not needed for Falkner — scope creep) · RecurrentPPO / PufferLib port (no failure traces to the optimizer).

---

## Agent 033 (ex PPO_CNN_10) — run log (2026-06-11, STOPPED at ~41M by 40M gate)

Smoke 5M clean (fps 2257→2700, all telemetry keys, checkpoints fire). Long run launched by user with
TensorBoard monitoring.

- **Gate 10M: FAIL** — reach_cherrygrove ≈ 0, policy near-random (entropy −1.86 of max 2.08), eval@10M
  wanders 245 tiles/4000 steps with 0% battles. No instrumentation bug. Learning SLOWER than Agent 023
  (which escaped New Bark by ~7M) — candidates: heavier obs (16ch + CombinedExtractor), MAX_STEPS 65k
  diluting signal. User decision: wait to 20M.
- **10→27M: it LEARNS** — reach_cherrygrove up to 0.32, egg_received_rate up to **0.14** (egg quest
  pickup finally happening, first time ever from start.state), in_battle 0.17, survives better.
- **27→41M: oscillation without consolidation** — egg_received bounces 0.13→0→0.06→0.03, cherrygrove
  0.14–0.31, visited_tiles keep rising (575-590) = exploration drifting to SAFE areas instead of the
  quest. **egg_delivered_rate: flat ZERO through 41M** — the delivery event was never experienced once,
  so it never entered the gradient. Structural: the ~600-step backtrack to Elm crosses already-visited
  tiles paying 0, with battle risk; the value function learns "going south with the egg = no income".
- 40M gate criterion met (delivery flat-zero) → REMEDY (Agent 034). Checkpoints kept:
  `runs/checkpoints/agent_033/` (5M…40M).

## Agent 034 (ex PPO_CNN_10b) — return-leg breadcrumbs + delivery +20, warm from Agent 033@40M (2026-06-11, LAUNCHED)

Reward-only remedy (obs unchanged → Agent 033 40M checkpoint loads, keeps Cherrygrove/pickup/combat skills):
- **RETURN_BREADCRUMBS** (rewards.py): per-episode latched map-ENTRY bonuses paid only while
  `egg_received AND NOT egg_delivered` — Cherrygrove +2, New Bark +2, Elm lab +2. Factorizes the
  impossible 600-step jump into learnable sub-goals. Gating makes cycling impossible (pays nothing
  before pickup, nothing after delivery). Idea: user's, matches the planned 25/40M no-go remedy.
- **Egg delivery weight 12 → 20** in `_event_score`.
- Synthetic tests: no-fire without egg ✓, +2 on entry with egg ✓, latched ✓, no-fire post-delivery ✓,
  pre-seed for egg-in-hand starts ✓.
- Config: RUN_NAME=Agent 034, INIT_FROM_CHECKPOINT=agent_033_39999936_steps.zip, budget 150M.
- **Watch**: nav/egg_delivered_rate must leave zero — that IS the run's purpose. Then the normal
  gate chain applies (reach_route31, reach_violet, in_battle ∈ [0.05,0.3], badge_rate).

**Run outcome (STOPPED ~55M):**
- 15M: pickup near zero — warm-start from the 40M checkpoint inherited the REGRESSED policy (pickup
  had already decayed 0.14→0.03 by 40M in Agent 033). Eval cross-check: not even the 25M/30M checkpoints
  reach Cherrygrove in 8k steps — the training reach metrics were INFLATED by 30-50k-step episodes
  (random-walk-with-bias, not a learned route).
- 50M: real recovery — reach_cherrygrove 0.09→0.55, egg_received back to 0.05-0.09 with rollout spikes
  at 0.5-1.0, ep_rew 1.27→1.71. But **egg_delivered still ZERO** (92M cumulative steps, zero delivery
  events ever — breadcrumbs verified correct in synthetic tests but never triggered in training).
- **USER EVAL DISCOVERY (watch mode, the decisive observation)**: the agent does HIT-AND-FLEE in every
  battle — one move, then RUN. ~20 battles per 2.5k steps, **0 won / all fled, Totodile stuck at lv 5**.
  Chip damage accumulates battle after battle → blackout. Corollary: with zero wins there is zero EXP,
  so the LEVEL REWARD NEVER FIRED in the entire filone — it was dead weight. Root cause: battles paid
  NOTHING (they only block tile income) → fleeing is optimal. (NOT the gym damage reward — that is
  map-gated to (10,7) and never fires on routes.)

## Agent 035 (ex PPO_CNN_10c) — COLD restart: + battle WIN reward, MAX_STEPS back to 32k (2026-06-11/12, LAUNCHED)

User decision: cold start (the 10/Agent 034 checkpoints carry the flee vice baked into the weights; cleaner
to learn battle engagement from scratch with the win reward active). Changes vs Agent 034:
- **Battle WIN reward** (rewards.py): +2.0 pre-scale on the battle falling edge with enemy HP VERIFIED
  at 0 (KO), our hp > 0. NO flee penalty (every battle penalty in history taught grass avoidance).
  CAPPED at 10 wins/episode (max 2.0 post ≈ 1/6 of story chain) — differs from the ruled-out Agent 010
  flat +15 (uncapped, not KO-verified). Grinding self-extinguishes twice: win cap + level saturation.
- **MAX_STEPS 65,536 → 32,768** (pokemon_env_cnn.py): the 65k experiment cost ~3-5× sample efficiency
  vs Agent 023 — fewer resets (less practice of the early corridor) and long zero-reward tails diluting
  the advantage signal. 32k is still 2.2× the optimal badge path.
- **Battle outcome telemetry**: env counts battles_won/fled/lost (falling-edge classification) →
  info → nav/battles_*_mean in TensorBoard; also shown in eval.
- **evaluate_cnn.py upgraded**: per-episode waypoint name, egg R/D, W/F/L, lead level, end cause
  (badge/death/truncated/step_cap), top-action distribution, and `--log` → JSONL in runs/eval_logs/
  (enables autonomous evals). Fixed pre-existing bug: with --max-steps the env was NOT reset between
  eval episodes (counters/tiles leaked across episodes).
- Gates (config.py): 10M reach_cherrygrove ≥ 0.6 AND battles_won_mean > 0 · 30M egg_received ≥ 0.1
  rising · 50M egg_delivered ≥ 0.2, reach_route31 > 0.1 · 100M badge_rate > 0.
- Note for a future cold restart if needed: drop SELECT (and maybe START) from the action space
  (8→6, ~25% exploration speedup; breaks checkpoints). Eval showed select pressed ~12% (uniform-ish
  entropy noise, not a learned habit) — not urgent.

**Run outcome (STOPPED at 5.8M by autonomous monitor): KAMIKAZE GRINDER.** The win reward worked
violently well — battles_fled 8.0→0.2, battles_won 0.6→4.9/ep — but a new local optimum emerged:
win 4-5 battles in the Route 29 grass, DIE (battles_lost ≈ 1.0/ep), respawn. ep_len collapsed
26k→2.5k, in_battle 0.45, hp 0.59, exploration frozen (~120 tiles, cherrygrove=0) and ep_rew was
RISING (1.29) on that cycle. Root cause: death at -1 pre (-0.1 post) was nearly FREE and — worse —
episode reset also RESETS THE WIN CAP, so dying refills the win budget. Death was the exploit.

## Agent 036 (ex PPO_CNN_10d) — COLD restart: death penalty -1 → -8 (2026-06-12, LAUNCHED autonomously)

Identical stack to Agent 035 with one surgical fix: **death penalty -8.0 pre-scale (-0.8 post ≈ 40% of the
max win budget)**. The kamikaze cycle becomes net-poor (+2.0 wins − 0.8 death over 2.5k steps) vs
surviving (+2.0 wins + tiles + story chain). Synergy with the existing heal reward (+2) should now
also teach healing. Cold start (the kamikaze vice was only 5.8M ≈ 35 min of compute — cheap to drop).
Watch at 10M: battles_won still > 0 (win behavior retained) AND ep_len recovering toward 15-25k
(no more suicide cycles) AND reach_cherrygrove ≥ 0.6.

**Run outcome (STOPPED at 10.1M by gate): kamikaze KILLED, but corridor still not learned.**
- Death penalty worked: ep_len healthy 27-30k, hp 0.8-0.9, no suicide cycles. Battle profile became
  risk-averse (fled 15-38/ep — a large TIME tax, ~1.5-3k steps/ep in flee menus; won 0.2-0.7).
- Healthy climb phase to ~7M (tiles 79→345, ep_rew −0.005→0.84) then **PLATEAU at 10M**: tiles flat
  ~380, ep_rew flat 0.79, reach_cherrygrove sporadic 0-0.08 (target 0.6), egg 0. Flat derivative =
  intervention condition met.
- **Elimination logic**: MAX_STEPS now equals Agent 023 (which had cherrygrove=1.0 at 7M with the same
  tile economy) and the corridor is still not learned → prime suspect is the OBS — specifically the
  visited-mask channel, never tested in isolation, and semantic NOISE whenever the screen shows
  battle UI / menus (8-45% of steps: the mask keeps painting overworld tiles on top of battle
  screens). Note: PWhiddy v1 did NOT use Dict obs — he blended exploration memory INTO the image
  with plain CnnPolicy.

## Agent 037 (ex PPO_CNN_10e) — MASK ABLATION, cold (2026-06-12, LAUNCHED autonomously)

Single change vs Agent 036: **visited-mask channel removed** (image back to 72×80×3; vector 11 and the
full Agent 036 reward stack unchanged — win +2 capped, death −8, breadcrumbs, event chain). Cold start
(obs change breaks checkpoints anyway). test_random_env_cnn updated and green.
- If Agent 037 reaches cherrygrove ≥ 0.6 by ~10M → the mask was the culprit (keep it out, or re-design it
  as in-image blending à la PWhiddy v1).
- If Agent 037 also plateaus → next suspect is the Dict/CombinedExtractor itself → test plain CnnPolicy
  (72×80×3, no vector) + fold critical bits (egg state) into the image as drawn pixels.

**GATE 10M: PASSED — THE MASK WAS THE CULPRIT.** reach_cherrygrove 0.652 (target 0.6 ✓, vs Agent 036's
flat 0-0.08 at the same age), reach_route30_gate 0.24, **egg_received_rate 0.121 already established
at 10M** (10/Agent 034 needed 25M+ for similar), tiles 589 (max 731), ep_rew 1.34 still climbing, full
survival (ep_len 31.7k, hp 0.86). The visited-mask channel — semantic noise on battle/menu screens,
4 extra stacked channels — was throttling CNN learning all along. **Ruled out: visited-mask as a
separate image channel.** If frontier memory is ever needed again, blend it INTO the RGB (PWhiddy v1
style), never as a parallel channel.
- Watch item (deferred): battles_fled 46-91/ep = huge time tax (~20% of episode in flee menus),
  battles_won only 0.12-0.23 — the win reward converts little at lv5-7. Revisit if the gym wall
  appears; do NOT touch while delivery is consolidating.
- Next: 30M gate (egg_received ≥ 0.1 rising ✓ already; expect first deliveries — breadcrumbs armed).

**15-20M: pickup becomes SYSTEMATIC (0.93-0.98), cherrygrove 0.95-1.0, ep_rew 2.38 — but delivery
still flat ZERO. Probe on the 20M checkpoint found the reason (STOPPED at 20M):** post-pickup the
agent parks INSIDE Mr.Pokemon's house (6.5k steps/ep!) + Route 30 north (4.5k) and NEVER goes south
— a **reward-ghost attractor**: the value function pins high value where the +8 egg event fired, and
with all local novelty exhausted there is no competing gradient. The latched breadcrumbs sit 600
unrewarded steps away → never experienced once in thousands of egg-in-hand episodes.

## Agent 038 (ex PPO_CNN_10f) — RETURN-POTENTIAL shaping, warm from Agent 037@20M (2026-06-12, LAUNCHED autonomously)

Reward-only (warm-start preserves the gold Agent 037 behavior: pickup 95%):
- **Potential-based return chain** while carrying the undelivered egg: map-transition reward
  k×(dist_old − dist_new) toward Elm, k=1.0 pre-scale, over the chain MrPokemon(5) → R30-north(4)
  → Cherrygrove(3) → Route29(2) → NewBark(1) → ElmLab(0). South +1, north −1, round trip = 0 net →
  UNHACKABLE (no profitable cycle), gradient lands exactly on the house door. Off-chain maps and
  egg-not-held / post-delivery states: no shaping (verified by 7 synthetic tests, all green).
- Latched breadcrumbs (+2 Cherrygrove/NewBark/ElmLab) kept on top as asymmetric pull.
- Expected sequence: house exit → southward chain (+0.5 post cumulative) → breadcrumbs (+0.6) →
  delivery (+2.0) → THE GATE OPENS → Route 31 waypoints take over.

**Run outcome (STOPPED at 13M): REWARD-DODGING — the agent stopped picking up the egg.** The
shaping moved it out of the house (probe @5M: 254 steps inside vs 6.5k in Agent 037 ✓) but pickup
collapsed 0.87 → 0.27: with the egg in hand the established north-dwelling habit bled −0.1 per
transition, and the agent discovered that NOT taking the egg ends the bleed entirely. **Lesson
(general): potential shaping is only unhackable WITHIN the shaped state — never attach negatives
to a state the agent can simply refuse to enter.**

## Agent 039 (ex PPO_CNN_10g) — Positive-only latched southward chain, warm from Agent 037@20M (2026-06-12, LAUNCHED)

Replaced the symmetric potential with **latched positive-only breadcrumbs over the full return
chain while carrying the egg**: R30-gate +1, Cherrygrove +2, Route29 +1, NewBark +2, ElmLab +2
(total 0.8 post for the full descent + delivery 2.0). No negatives → no pickup-avoidance; latched
→ no cycling; gated on carrying → outbound pays nothing. 3 synthetic tests green (chain total 0.8,
re-walk pays 0, no-egg pays 0). Warm from Agent 037@20M (the pre-contamination checkpoint with pickup 95%).

**Run outcome (STOPPED at 8.1M): pickup HELD at 0.97 (no avoidance relapse ✓) but still bouncing in
the northern pocket** (probe @5M: R30-north 4.2k + MrPokemon house 3.4k steps post-pickup), ep_rew
only +0.08 over the Agent 037 baseline → the southward chain is barely being experienced. Root economics:
**the southern backtrack pays ZERO tile reward (all visited outbound)** — the agent's one proven
navigation driver is exhausted exactly where it is needed; one-shot breadcrumbs (0.1-0.2) don't
compete with habit.

## Agent 040 (ex PPO_CNN_10h) — TILE-NOVELTY RESET on egg pickup, warm from Agent 039@5M (2026-06-12, LAUNCHED)

Env-side fix that re-uses the proven engine: **when the egg lands in hand, clear the EPISODE
visited-tiles set** (one line in step(), one-shot per episode — the egg flag is latched, unfarmable;
savestates already holding the egg are seeded so they don't trigger it). The whole world becomes
fresh again: the southern corridor pays dense tile income on the return (~400 tiles ≈ 0.4 post) on
top of breadcrumbs (0.8) and delivery (2.0); the northern-pocket re-earn is small change (~0.2).
Expected: post-pickup the tile-novelty gradient — which the agent demonstrably follows — now points
in EVERY direction including south, and the breadcrumb chain + delivery tips the balance south.

**Run outcome (STOPPED at 8M): the reset became a "novelty refresh button".** Emerging strategy:
milk all tiles pre-pickup, take the egg (reset), RE-MILK the nearby northern pocket — double
harvest without ever going south. Pickup even got PROCRASTINATED (batches at 0.0 = episodes
postponing it past truncation; oscillating 0.27-1.0, recent mean 0.60). ep_rew 3.03 ↑ but
delivery still 0. Battles hypothesis disproven (won 0.065 — not converting). Lesson: an
unconditional novelty refresh subsidizes the NEAREST area, not the intended direction.

## Agent 041 (ex PPO_CNN_10i) — DIRECTIONAL tile reset, warm from Agent 040@5M (2026-06-12, LAUNCHED autonomously)

Refinement: at pickup, clear all visited tiles EXCEPT the northern pocket maps (Route 30 north
(26,1) + Mr.Pokemon's house (26,10)). Post-pickup the north pays ZERO (still marked), the only
dense income is the SOUTHERN corridor (fresh tiles ~0.6-0.8 post) + breadcrumbs (0.8) + delivery
(2.0). Kills the double harvest AND the procrastination (pre-milking no longer doubles), keeps
pickup strictly profitable. Warm from Agent 040@5M (pickup 1.0, pre-exploit).

**Run notes (STOPPED at 9.4M for instrumentation):** healthy V-shaped adaptation — pickup dipped to
0.25 (unlearning the dead north re-milk) then recovered to 0.55-0.85; ep_rew V too (2.83→2.02→2.55);
some episodes explore 750+ tiles post-reset (fresh territory somewhere). Delivery still 0 — but we
were FLYING BLIND on the only question that matters: how far south does the front reach?

## Agent 042 (ex PPO_CNN_10j) — Same as Agent 041 + return-front INSTRUMENTATION (2026-06-12, LAUNCHED autonomously)

No reward change. Added `nav/return_progress_mean`: southernmost chain map reached per episode while
carrying the egg (0=north pocket, 1=R30, 2=Cherrygrove, 3=Route29, 4=NewBark, 5=Elm lab). Warm from
Agent 041@5M. The next fix (if needed) will target the EXACT link where the front stalls:
- front ≈ 0-1 → discovery problem → ent_coef 0.03→0.05
- front ≈ 2-3 (reaches Cherry/R29, then turns back) → deepen mid-chain breadcrumbs
- front ≈ 4-5 without delivery → the Elm-lab DIALOG is the wall (needs A-press at the right spot)

**Readout @4.2M (STOPPED at 4.8M):** front mean 0.83 with pickup 0.55 → ~half of egg episodes touch
CHERRYGROVE(2) then turn back; none reach Route29(3). The front advances one link per run (house→R30
in Agent 038-i, R30→Cherry now). Decision-tree branch: mid-chain stall → deepen downstream pull.

## Agent 043 (ex PPO_CNN_10k) — Mid-chain deepening, warm from Agent 042@5M (2026-06-12, LAUNCHED autonomously)

Reward-only: ROUTE_29 breadcrumb +1→+3, NEW_BARK +2→+3, egg delivery 20→30 (thunderclap when it
first lands). Front telemetry continues. Expected: Cherry→R29 link conquered next, then the chain
accelerates (NewBark is 1 transition from R29, the lab 1 more, both heavily paid).

**Run outcome (STOPPED at 12.4M): WARM-RESTART CHURN EROSION.** The value function never settled
(explained_variance bouncing negative, even -5.2), ep_rew bled 3.1→1.95, pickup degraded to ~0.2
with isolated 1.0 spikes (front spike to 2.0 = the capability survives but is rarely expressed).
**Meta-lesson: 5 consecutive warm-restarts with changed rewards (f→g→h→i→j→k) each re-fit the value
from a stale fit and eroded the behavior — warm-starting is cheap ONCE, but chaining it compounds
churn faster than learning.**

## Agent 044 (ex PPO_CNN_10L) — COLD with the full final stack (2026-06-12, LAUNCHED autonomously)

End of the warm chain. One coherent value function from step 0, every signal present from the start:
- Obs: image 72×80×3 (NO mask) + 11-vector (egg bits, trainer counts).
- Rewards: event-dominant chain (egg 8, delivery 30, trainers +5, badge 30, rival 3) · win +2
  KO-verified cap 10 · death −8 · level saturation 15 · waypoints +2 · Mr.Pokemon house +2 gated ·
  RETURN: directional tile reset at pickup (north pocket stays spent) + southward latched chain
  R30+1 / Cherry+2 / R29+3 / NewBark+3 / Lab+2.
- Reference timeline: Agent 037 (cold, FEWER aids) hit pickup 95% by 15-20M. Gates: 12M cherrygrove ≥0.5;
  20M pickup ≥0.6; 30M front ≥2 and first delivery expected.

**Run outcome (STOPPED at 20.4M):** learning ORDER inverted vs Agent 037 (battles pay now → fight phase
first: W/L up to 50:1, in_battle 0.46, hp 0.39 — real combat competence for the first time ever).
Navigation emerged mid-run (cherry 0.77, pickup 0.28 @15-17M, ep_rew 3.5) then **the optimizer
overshot its own best phase**: 12 rollouts of flat-zero navigation, ep_rew halved to 1.75. Chronic
clip_fraction 0.25-0.30 on clip 0.1 = persistent overshoot signature. Eval@15M: fights superbly
(18W/0L) but 0/3 cherrygrove in 12k-step episodes — navigation existed only as slow drift.

## Agent 045 (ex PPO_CNN_10m) — 65k episodes + halved lr, warm from Agent 044@15M (2026-06-12, LAUNCHED autonomously)

**Structural re-think — the delivery may have been time-starved, not reward-starved:** in ~170M
steps across 9 runs the delivery was never EXPERIENCED once. PWhiddy's agent solved the analogous
Oak's-Parcel backtrack with 163,840-step episodes (~150k steps of post-pickup wandering); ours had
~20k at MAX_STEPS 32k. The 32k cut (Agent 035) was justified by the Agent 035/Agent 036 slowness — later PROVEN to be
the visited-mask's fault (Agent 037 ablation) and never re-tested. Changes:
- **MAX_STEPS 2^15 → 2^16 (65,536)**: ~45k post-pickup steps for delivery discovery.
- **lr 1.5e-4 → 8e-5** (train_cnn now overrides the checkpoint lr on warm-load via custom_objects):
  stop overshooting good policies.
- Warm ONCE from Agent 044@15M (battle competence + pickup in repertoire). Full reward stack unchanged.

**Run outcome (STOPPED at 20M): FIGHT ATTRACTOR LOCK.** The inherited fight-phase policy stayed
locked: 12 wins/ep flat through 20M, hp 0.24-0.38, tiles ~190, quest at zero, cherry flickers
≤0.13. The stability of lr 8e-5 (clip_fraction 0.15 ✓) also made the phase STICKY — the 2.0-post
battle ceiling kept absorbing the gradient. Disk artifacts renamed to agent_045.

## Agent 046 — Win cap 10→5, warm from agent_045@20M (2026-06-12, LAUNCHED autonomously)

Reward-only: the battle win cap halves (max 1.0 post/episode). Anti-flee is deeply learned (12
wins/ep); freeing 1.0 of per-episode income pressure should redirect the gradient to tiles + the
egg quest, with lr 8e-5 keeping the transition gentle. 65k episodes unchanged — the delivery-time
hypothesis still stands once the quest behavior re-emerges. First run under the agent_NNN naming.
Gates: 8M nav re-emergence (cherry ≥ 0.3); 16M pickup > 0.3; then delivery watch.

**Run outcome (STOPPED at 8M):** nav flickered (cherry peak 0.45 @5M, pickup 0.08) then receded
again (8M gate: cherry 0.19, pickup 0, won still 10.3). The fight habit decays too slowly at lr
8e-5 and nav waves cannot consolidate. Six reward-side attempts on the same link = diminishing
returns; switching solution class.

## Agent 047 — EXPERIENCE INJECTION via return-leg envs (2026-06-12, LAUNCHED autonomously)

Go-Explore / puffer state-sharing style: **3 of 12 envs start from `mid_route30.state`** (verified:
Cherrygrove (26,3), egg in hand, NOT delivered, Totodile lv7 — the lab is 2 transitions away).
Those envs will EXPERIENCE the delivery within minutes, finally injecting the never-seen event
(+30 → 3.0 post) into the gradient; the bet is that the value function then prices the southward
chain for the start.state envs too (same corridor, same policy).
- Differs from the ruled-out CNN_7 mixed curriculum: same quest corridor (same visual domain), and
  **metrics split by origin** — `nav/*` = start.state envs only, `navret/*` = return-leg envs —
  so start-state regressions cannot hide (InfoLoggerCallback rewritten).
- Win cap 5, 65k episodes, lr 8e-5 unchanged. Warm from agent_046@5M.
- Watch: `navret/egg_delivered_rate` must go ≥0.5 fast (sanity that delivery works mechanically);
  then `nav/return_progress_mean` and `nav/egg_delivered_rate` = the TRANSFER signal; then the
  cascade (route31 → violet → gym → badge) from either origin.

**Run outcome (STOPPED at 5M): the policy prior beats the injection point.** The return envs went
NORTH (navret/ep_max_waypoint=2 = R30 gate — their trained drift) and delivered ZERO times in ~60
episodes despite the lab being 2 transitions south. The northbound habit is so strong that even
exploration noise never produces the southward traversal.

## Agent 048 — GRADIENT LADDER along the return chain (2026-06-12, LAUNCHED autonomously)

Inject at the DOORSTEP: 8×start + 2×mid_route30 (Cherrygrove, egg) + **2×before_elm_delivery
(INSIDE Elm's lab, egg deliverable — verified (24,5), egg=1, delivered=0)**. The lab envs deliver
by noise within steps → the +30 event finally enters the gradient; the mid envs bridge the value
propagation southward; the start envs harvest the transfer. Ladder = reverse curriculum WITHIN one
run, with origin-split metrics. Warm from agent_047@5M.

**Run outcome (STOPPED at 15M): RUNG 1 CONQUERED — FIRST DELIVERIES IN PROJECT HISTORY** (~250M
cumulative steps in). At 1.25M the lab envs delivered for the first time; by 10M they deliver in
~100% of their episodes (navret batches alternate 1.0/0.0 = lab-env vs bridge-env episode endings).
BUT the Cherrygrove→lab gap (2 transitions) stayed unbridged for 10M — the bridge envs kept walking
north. The ladder needs a denser rung.

## Agent 049 — Denser ladder: + New-Bark-with-egg rung (2026-06-12, LAUNCHED autonomously)

Created `saves/newbark_egg.state` PROGRAMMATICALLY (scripted guided walk out of the lab door with
live RAM feedback — 10 moves; verified (24,4), egg=1, delivered=0). New ladder:
7×start + 2×Cherrygrove(rung 3) + **2×NewBark-egg(rung 2, 1 hop from the lab)** + 1×lab(rung 1,
consolidated). Warm from agent_048@15M. Expected: rung 2 bridges by noise (single transition),
then rung 3's 1-hop gap to rung 2 becomes learnable, then the start envs inherit the full chain.

**Run outcome (STOPPED at 12M): rung 2 BRIDGING** — ret_deliv rose to 0.28-0.44 (baseline 0.20 if
only the lab env delivered): the NewBark-egg envs deliver in ~half their episodes. Mechanics
proven; pace is lr-8e-5 slow. Decision: extend the ladder FORWARD instead of waiting hours of
sequential discovery for the post-gate corridor.

## Agent 050 — FULL-ROUTE LADDER (2026-06-12, LAUNCHED autonomously)

Rungs across the entire quest: 6×start + Cherry-egg + NewBark-egg + lab + **crossing (post-gate)** +
**violet_city** + **violet_city_gym (inside Falkner's)**. Every segment gets simultaneous experience;
the gym rung can win the badge early (battle competence W/L ~50:1 is in the warm-start) putting the
+30 badge into the gradient from the start. Higher segregation risk on the late, visually distinct
states — accepted with eyes open: origin-split metrics expose any start-env regression immediately
(none observed across 047-049). Warm from agent_049@10M.

**MILESTONE @0.8M: the whole route lit up simultaneously** (route31 1.0 batches, violet/gym
registering, deliveries streaming).
**MILESTONE @2.0M: FALKNER BEATEN — FIRST ZEPHYR BADGE IN PROJECT HISTORY** (gym-rung env episode
ended with badge_rate 1.0 at step 1,990,656). Every reward in the quest chain has now been
EXPERIENCED. What remains is consolidation per rung and the backward stitching to start.state.

**Run outcome (STOPPED at 12.6M): mission accomplished + the predicted cost.** Badge ×6 at steady
~1.5-2M cadence, deliveries systematic (0.83-1.0). But start-env nav collapsed exactly as the CNN_7
precedent warned (nav_cherry thirds: 0.10 → 0.125 → 0.000) — the ladder envs' income feast pulled
the policy away from the start distribution. The ladder did its job (all rewards in the gradient);
time to give the weight back.

## Agent 051 — REBALANCE: 8×start + 4 anchors (2026-06-12, LAUNCHED autonomously)

8×start.state + one anchor per critical segment (Cherry-egg / NewBark-egg / crossing /
violet_city_gym — the lab rung retired, its transition is covered by the NewBark anchor). The
anchors hold the value landscape (badge keeps firing, deliveries keep flowing) while the start
envs get 2/3 of all samples to re-find pickup and stitch the chain — this time pulled by a value
function that KNOWS the quest pays ~14 post-scale. Warm from agent_050@10M.
Watch: nav_cherry recovery → nav pickup → nav delivery (transfer) → nav badge = THE GOAL.

**Run outcome (STOPPED at 10.8M): even 8/4 wasn't enough — anchors still captured the policy.**
Brief recovery (cherry + one pickup batch @4.8M) then full nav collapse. Probe @10M: start envs
exit New Bark instantly then FARM ROUTE 29 BATTLES 99% of the time (14W+28F per 10k steps, egg
never taken). The battle attractor — reinforced by the anchors' combat gradient — captures the
start policy at the first grass.

## Agent 052 — PURE START finishing run + win cap 2 (2026-06-12, LAUNCHED autonomously)

All 12 envs on start.state (the real distribution, full attention), win cap 5→**2** (0.4 post —
battle income stops competing with the ~14-post quest; the battle COMPETENCE stays in the weights
for the gym). The quest value landscape imprinted by 048-051 decays slowly at lr 8e-5 — the bet is
the start envs stitch the chain before it fades. Warm from agent_051@10M.
Milestone events armed: pickup recovery → delivery from start → Route31 → gym → BADGE FROM START.

**Run notes (STOPPED at 40M): the OSCILLATION signature, again.** Strong recovery through 30M
(cherry 0.28→0.56→**0.90**, pickup 0.17) then recession by 40M (cherry 0.28, pickup 0.08). With
ent_coef 0.03 the policy VISITS the right behaviors but never LOCKS them — a limit cycle of
episode-novelty income + diffuse policy. Peak checkpoint saved at 30M.

## Agent 053 — COMMIT run: ent_coef 0.03→0.01, warm from 052@30M peak (2026-06-13, LAUNCHED)

The one knob never turned the right way: **entropy DOWN** (to PWhiddy's 0.01) — after ~300M steps
of exploration it is time to let the policy commit and exploit the stitched route. Warm from the
best stitching state ever (052@30M, cherry 0.90 era). train_cnn now overrides BOTH lr and ent_coef
on warm-load (PPO.load restores them from the checkpoint otherwise). Risk (premature convergence)
is mitigated by starting AT the peak: what it would converge onto is the desired behavior.

**Run outcome (STOPPED at ~84M): SEGMENT 1 CONSOLIDATED, delivery still walled.** ent_coef 0.01 did
exactly its job — `reach_cherrygrove` LOCKED at 1.0 and `egg_received_rate` at 1.0 across the whole
back half (the most stable, highest segment-1 policy in project history — no oscillation, the 052
cherry-0.90↔0.28 limit cycle is gone). But `egg_delivered_rate` stayed flat **ZERO across all 1097
logged rollouts**; `return_progress_mean` sat at **1-2** (carries the egg back only to Route 30 /
Cherrygrove, then turns around) with sporadic spikes to 3 (Route 29) and 4 (New Bark) — the southward
traversal HAPPENS on exploration noise but never consolidates. `battles_fled` 86-102/ep = a large
time tax; hp ~0.8 (survives). Root cause confirmed: post-pickup the committed policy re-milks the
**re-armed Cherrygrove** (053's reset spares only the north pocket) instead of pushing south.
Checkpoints 5M…80M saved in `runs/checkpoints/agent_053/`.

## Agent 054 — RETURN-CORRIDOR pickup reset, warm from 053@80M (2026-06-13, LAUNCHED autonomously)

ENV-ONLY change (single variable for clean attribution vs 053): the egg-pickup tile-novelty reset
(`pokemon_env_cnn.py`) now re-arms **ONLY the southern delivery corridor** (Route 29 → New Bark →
Elm's lab); the north pocket, Route 30 gate, AND **Cherrygrove** all stay SPENT.
- **Diagnosis**: 053's reset re-armed everything-but-the-north-pocket → Cherrygrove stayed re-armed →
  the agent re-milked it post-pickup and the return front stalled there (`return_progress` 1-2,
  delivery 0). Keeping Cherrygrove spent moves the nearest fresh-tile income to **Route 29** — a
  directional pull through the exact Cherrygrove→Route29 link where the front stalls, along a corridor
  the policy already traverses northbound at 100%. Removes the Cherrygrove re-milk attractor that
  Agent 041's "all-except-north-pocket" reset accidentally created.
- **Safety**: pre-pickup behavior is unchanged → pickup stays strictly profitable (no pickup-avoidance,
  the 10f/Agent 038 failure mode). Latched on the egg flag, one-shot per episode, unfarmable.
- Warm from 053@80M (pickup 100%). ent_coef 0.01, lr 8e-5, win cap 2, MAX_STEPS 65k, 12×start.state —
  all IDENTICAL to 053.
- **Watch / gates**: `nav/return_progress_mean` must climb past 2 toward 5; `nav/egg_delivered_rate`
  must LEAVE ZERO (the run's whole purpose — first start.state delivery EVER). Then the gate opens:
  reach_route31 → reach_violet → reach_gym → badge_rate. Gate checks: ~15M return_progress rising past
  2 AND egg_received held ≥ 0.6; ~30M first deliveries; ~50M egg_delivered ≥ 0.2 and reach_route31 > 0.1.

**MILESTONE @~17.5M: FIRST EGG DELIVERY FROM START.STATE IN PROJECT HISTORY.** `egg_delivered_rate`
hit 1 in a rollout (latched ELM_BIT, real) — the ~600-step backtrack that was NEVER experienced from
start across 350M+ cumulative steps. `return_progress_mean` now reaches 2-3 (Route 29) regularly vs
053's flat 1-2 — the directional reset (Cherrygrove kept spent) does pull the front south as designed.
**Run outcome (STOPPED at ~25M): delivery REACHABLE but NOT consolidated, and the reset cost pickup.**
Windowed means (40-sample windows) cut through the per-rollout noise:
- `return_progress_mean`: rose to a peak **1.33** around 17-19M (the delivery region), then RECEDED to
  **0.54** by 25M. Only ONE delivery total (the 17.5M spike) — a lucky exploration event, not a learned route.
- `egg_received_rate`: stuck at **0.55-0.74**, NEVER recovered to 053's solid 1.0 → the reset REGRESSED pickup.
- `ep_rew_mean`: oscillated 2.5-3.2, below 053's 3.95. `reach_route31` flat 0 (gate never crossed).
- **Diagnosis**: removing Cherrygrove's re-armed tiles also removed the post-pickup income that was
  PROPPING UP the pickup value in 053 — so picking up became slightly less attractive (mild reward-dodge),
  while the southward income past spent-Cherrygrove sits behind a zero-income gap the committed (ent 0.01)
  policy won't reliably cross. The Cherrygrove→Route29 transition is a specific DOOR problem that needs a
  DIRECTIONAL VALUE signal, not more global novelty/entropy. (Confirms: tile-novelty is a local gradient —
  it can't pull the agent across a barren gap to a fresh zone beyond. Same wall as Agent 026/040.)
- **Kept**: the directional reset is still correct (it made deliveries reachable for the first time) —
  carried into 055. **Ruled out**: directional reset ALONE (no delivery-value source) — reaches delivery on
  noise, can't consolidate it, and degrades upstream pickup.

## Agent 055 — DELIVERY-VALUE ANCHOR (11×start + 1×lab), warm 053@80M (2026-06-13, LAUNCHED autonomously)

Synthesis of the pieces that each gave partial signal. Keeps 054's directional corridor reset (env); adds
ONE `before_elm_delivery` anchor (verified: Elm lab (24,5), egg in hand, undelivered, lead lv8 — delivers
within steps). Rationale: the anchor delivers EVERY episode → the +30 (3.0 post) delivery value enters the
SHARED value function → the start envs' "at Cherrygrove with egg, facing south" state acquires HIGH value =
exactly the directional gradient at the door that 054 lacked; the directional reset then pays dense income
when the start policy follows that gradient south.
- **Why an anchor when 047-051 segregated**: this is ONE minimal, low-income, visually-familiar (lab ≈ New
  Bark area) anchor — well below the collapse threshold (Agent 048 ran 4 anchors WITHOUT start-nav collapse;
  collapse needed 6 diverse high-income rungs incl. gym/violet at 050/051). It sits on the strongest start
  policy ever (053 pickup 1.0, resists segregation) + the NEW directional reset (a landing pad for the
  transferred value). Origin-split metrics (`nav/*`=start, `navret/*`=anchor) expose regression within ~10M
  → ABORT if `nav/egg_received_rate` collapses below ~0.5.
- Warm from CLEAN 053@80M (not 054's pickup-sagged weights). ent 0.01, lr 8e-5, win cap 2, MAX_STEPS 65k.
- **Watch / gates**: navret/egg_delivered_rate ≈ 1 fast (anchor delivers — sanity); then the TRANSFER
  signals: nav/return_progress_mean past 2 AND nav/egg_delivered_rate leaving zero, with
  nav/egg_received_rate HELD ≥ 0.8. Then reach_route31 → violet → gym → badge.

**Run outcome (STOPPED at ~10M by the anchor-delivery criterion): the ANCHOR ITSELF NEVER DELIVERED.**
`navret/egg_delivered_rate` = **0 across all 14 anchor rollouts**. The anchor reaches the lab
(`return_progress`=5) but `ep_max_waypoint`=2 — it walks NORTH out of the lab carrying the undelivered
egg and never presses A at Elm. The committed 053 policy's northward habit beats the injection point
(exactly the Agent 047 failure), and unlike Agent 048 (which learned the delivery A-press by 1.25M from
a LESS-committed base) the ent-0.01 053 base won't explore it even by 10M. `nav/egg_received` (start)
sagged to 0.3-0.6 (the reset's pickup cost). **Lesson: a single low-entropy anchor can't inject delivery
value if the policy prior won't perform the delivery action — and ent 0.01 is too committed to discover it.**

## Agent 056 — ENTROPY SCHEDULE (0.03→0.01) + 2 lab anchors + directional reset, warm 053@80M (2026-06-13, LAUNCHED autonomously)

The genuinely-missing lever, isolated by the 053/054/055 through-line: **a FIXED low ent_coef (0.01) is
too committed to LEARN any new long-range behavior (the delivery A-press, the southward backtrack), but a
FIXED high ent_coef (0.03, Agent 052) oscillates and never locks segment 1.** No prior run used a SCHEDULE.
- **New code**: `EntCoefScheduleCallback` (train_cnn.py) linearly anneals ent_coef 0.03→0.01 over the
  first 50M steps, then holds 0.01 (SB3 reads model.ent_coef as a float each update, so a per-rollout
  mutation works; lr supports schedules natively, ent_coef does not). Explore high → commit low.
- **Curriculum**: 10×start + **2**×before_elm_delivery (055 used 1 and it never delivered; 2 + the higher
  early entropy should find the delivery A-press fast, like 048's 2 lab anchors did). Keeps the directional
  reset (env). 2 anchors is still below the 050/051 collapse threshold; nav/ vs navret/ split guards start.
- Warm from CLEAN 053@80M (pickup 1.0); the schedule re-introduces exploration ON TOP of the strong
  segment-1 base, so the policy learns delivery+south without discarding pickup. lr 8e-5, win cap 2, 65k.
- **Watch / gates**: ~10M navret/egg_delivered_rate ≈ 1 (anchors now deliver — the 055 failure fixed);
  ~25M nav/egg_delivered_rate leaving zero (TRANSFER) with nav/egg_received held ≥ 0.6; ~50M (ent fully
  annealed) the route locks: reach_route31 → violet → gym → badge. ABORT if nav/egg_received → ~0.

**Run outcome (STOPPED at ~5M): anchors STILL didn't deliver — confirming it's the BASE, not entropy.**
At ent 0.028 the 2 anchors showed the SAME failure as 055: `navret/egg_delivered_rate`=0, `return_progress`=5
(at lab) but `ep_max_waypoint`=2 (walks north out of the lab). Identical mechanism across ent 0.01 (055)
and 0.028 (056). Meanwhile the higher entropy DEGRADED start pickup further (`nav/egg_received` 0.15-0.39 vs
055's 0.3-0.6). **Decisive lesson: the 053-LINEAGE base policy has no delivery action in its repertoire —
its learned behavior at the lab is "exit building, go north", so its anchors leave before pressing A at Elm,
at ANY entropy. Entropy only adds noise around the dominant (northward) action; it can't synthesize a missing
skill.** The entropy-schedule CODE works (train/ent_coef_sched annealed 0.03→0.0281 by 5M) and is kept.

## Agent 057 — BASE SWITCH to agent_050 (delivery+badge-capable), warm @10M (2026-06-13, LAUNCHED autonomously)

Single variable vs 056: warm from **agent_050@10M** instead of the 053 lineage. agent_050 was the FIRST
policy to deliver the egg AND beat Falkner (badge at 2M) — the delivery+badge skill is IN its weights, so
its anchors deliver immediately (fixing the 055/056 anchor-never-delivers failure at its root). Keeps all of
056's machinery: directional reset (env), entropy schedule 0.03→0.01, 10×start + 2×lab anchors, win cap 2,
lr 8e-5, 65k.
- **The trade**: 050's START pickup is weak (it segregated — start envs farmed Route 29 battles, 051 probe).
  But pickup is the EASIER half to recover (053 re-learned 100% pickup from this very lineage) and the
  directional reset + entropy schedule target exactly that; the HARD half (delivery, never-from-start) is
  now present in the weights and kept alive by the anchors.
- **Differs from 051 (warm 050 + 4 anchors, which collapsed)**: only 2 anchors (not 4), win cap 2 (not 5 —
  defuses the Route-29 battle-farm attractor that captured 051's start envs), + the NEW directional reset
  (a non-battle southward gradient for start) + entropy schedule. nav/ vs navret/ split guards segregation.
- **Watch / gates**: ~10M navret/egg_delivered_rate ≈ 1 (anchors deliver from this base — sanity); start
  pickup recovering (nav/egg_received climbing back toward 0.6+); ~30M nav/egg_delivered leaving zero
  (TRANSFER) and reach_route31 > 0; then violet → gym → BADGE FROM START. ABORT if nav/egg_received → ~0
  (segregation) or by ~15M anchors still 0 (deeper problem than the base).

**Run outcome (STOPPED at ~10M): base switch FIXED anchor delivery, but the START SEGREGATED (the 051 wall).**
- The 050-base anchors DELIVER (`navret/egg_delivered_rate` 0.87-1.0) — diagnosis confirmed: 055/056's failure
  was the 053 lineage's absent delivery skill, not entropy. Value injection works from this base.
- BUT `nav/egg_received` (start) stayed **pinned at 0.00 for 5M straight** (one 0.07 blip) — the start envs
  never escaped 050's battle-farm basin (battles_fled 186/ep) to even pick up the egg, so the injected
  delivery value never transferred. Same segregation as 051: the anchors' rich income captures the shared
  policy. ep_rew 1.74. Win cap 2 + directional reset + entropy schedule did NOT prevent it.
- **CONCLUSION (anchors definitively ruled out)**: across 047-051, 055, 056, 057 — every fixed
  anchor/curriculum mix segregates the start policy, at every base/entropy/anchor-count/win-cap, with or
  without the directional reset. The pickup-policy and delivery-policy do not coexist in one shared CNN
  under a fixed curriculum. Only a START-CONTINUOUS reset distribution (Go-Explore frontier) could avoid
  the visual island — that is a larger build, flagged for the user.

### Eval — agent_050@10M from violet_city_gym.state (2026-06-13)
The "badge-winning" 050 does NOT robustly win the badge in eval. DETERMINISTIC: degenerate (stuck, 10 tiles,
0 battles, repeats one action pattern). STOCHASTIC @6k steps: 0/10 badge, 6 wins/0 losses total, 143 tiles.
STOCHASTIC @25k steps: **0/6 badge**, but Battles W/F/L = **288/38/1** (~48 wins/ep!), in_battle 34%, lead
lv 17-18, 2/6 episodes end in death. → 050 is a WILD-BATTLE GRINDER: from the gym state it roams the Violet
routes winning dozens of wild battles and leveling up, but never completes the Falkner fight. The 050 training
badges (×6 over ~12M steps) were RARE stochastic events, NOT a learned capability. **NO checkpoint robustly
beats Falkner even from 2 steps inside the gym — the gym fight is an UNSOLVED SECOND WALL, separate from the
delivery wall.** (Reframes project status: the badge has been TOUCHED ~6× ever, never made reliable from any state.)

## Agent 058 — PURE-START + entropy schedule + directional reset, warm 053@80M (2026-06-13, LAUNCHED autonomously)

The cleanest UNTRIED from-start lever, with NO segregation risk (no anchors). Warm from 053@80M (pickup 1.0)
+ directional corridor reset (env) + entropy schedule 0.03→0.01 (the lever 054 lacked — it was fixed at 0.01
and couldn't explore the southward path) + PURE 12×start, win cap 2, lr 8e-5, 65k.
- **Bet**: high early entropy (0.03) lets the pickup-100% policy EXPLORE the reset-rewarded southward corridor
  and DISCOVER delivery on its own; annealing to 0.01 then LOCKS pickup+delivery together. Relies on the start
  policy's own exploration + the reset gradient, not on value injection (which requires anchors that segregate).
- **vs 054** (053+reset, FIXED ent 0.01): the schedule adds the exploration 054 couldn't do. **vs 056**
  (053+reset+anchors+schedule): drops the anchors (which segregate and, on 053-base, didn't even deliver).
- **Watch / gates**: ~15M nav/return_progress climbing past 2 with nav/egg_received HELD ≥ 0.6 (pickup must
  survive the high-entropy phase); ~30M nav/egg_delivered leaving zero; ~50M (annealed) route locks. ABORT if
  pickup craters < 0.4 (entropy de-locking segment 1 without compensating delivery gains).

**Run outcome (STOPPED at ~9.8M for the architectural pivot): delivery FLAT ZERO, the entropy schedule
DE-LOCKED pickup — confirming the pure-start reward/entropy toolbox is exhausted for Phase 1.**
- `nav/egg_delivered_rate` = **0 across the entire run** (never delivered once in ~9.8M steps).
- `nav/egg_received_rate` oscillated **0↔1** (no longer 053's solid 1.0) — the rising entropy (0.01→0.026)
  de-committed the pickup habit without buying any delivery. This is the predicted ABORT signal (pickup
  de-locking with no compensating delivery gain).
- `nav/return_progress_mean` peaked at **2** (Route 29) and never reached the lab (5); `ep_rew_mean` ~2.2
  (below 053's 3.95). `nav/reach_route31` flat 0 (gate never crossed).
- **Root cause (the through-line of 053→058)**: the delivery is a MISSING ACTION (the Elm A-press) plus a
  ~600-step backtrack the start policy reaches too rarely to ever practice. Entropy only adds noise around
  the dominant northward action — it cannot SYNTHESIZE a never-performed action (the 055/056 lesson, now
  re-confirmed at the policy level on a pure-start base). Reward shaping (breadcrumbs), the directional
  reset, and the entropy schedule are ALL in the code and ALL insufficient.
- **Lesson / ESCALATION TRIGGER**: every from-start reward/entropy/reset lever is now deployed and Phase-1
  delivery is still flat 0 after six runs (053–058), with anchors definitively ruled out (segregation,
  047–051/055–057). The only remaining hypothesis is architectural. Per the operating protocol's escalation
  rule, STOP twiddling and write the design proposal below. **Do NOT launch agent_059 as another pure-start
  reward/entropy run** — it would re-confirm a known wall.

---

## DESIGN PROPOSAL — Go-Explore / Frontier-Reset (START-CONTINUOUS) — ESCALATION, PAUSE for human review (2026-06-13)

> Status: PROPOSAL ONLY. No code written, no run launched. Awaiting human approval before the build.

**Why now.** Six runs (053–058) attacked Phase-1 delivery from `start.state`; `nav/egg_delivered_rate` left
zero exactly once (054 spike, unconsolidated) and was flat 0 in the cleanest pure-start run (058). Anchors are
ruled out (segregation). The reward/entropy/reset toolbox is fully deployed and exhausted. Both project walls
(delivery, the Falkner fight) share ONE root cause: a deep sub-skill is never PRACTICED in-distribution because
the start policy reaches the frontier state too rarely to discover the missing action there. This is the
architectural lever the brief names as "likely required" — and it is the SAME unlock for both phases.

**Core idea.** Stop resetting envs only from fixed `.state` FILES. Reset a fraction of envs from PyBoy states
**harvested from the start policy's OWN recent trajectories**, sampled toward the frontier. Because those reset
states ARE the policy's own visited cells (identical pixels/distribution), there is NO visual island to
segregate — the agent practices "short backtrack from where I actually am, then press A at Elm" and, in Phase 2,
"the multi-turn Falkner fight from where I actually walked in," repeatedly and in-distribution. This is the one
reset distribution the project's segregation rule permits (it is continuous with the start trajectory, not a
foreign scene).

**Design in THIS codebase:**
1. **Cell key** (coarse hash bucketing distinct frontier positions):
   `(map_bank, map_number, local_x//K, local_y//K, egg_received, egg_delivered, gym_trainers_beaten)`.
   Including the story flags is what makes "Cherrygrove WITH egg, undelivered" a distinct frontier cell from
   "Cherrygrove without egg" — exactly the state the backtrack must start from.
2. **Archive**: bounded map `cell → (pyboy_state_bytes, visit_count, best_score)`. PyBoy supports
   `save_state(BytesIO)` / `load_state(...)`; we already call `pyboy.reset()` (a load) — we add in-rollout
   snapshotting. Bound to N cells (frontier-weighted eviction) to cap disk/RAM.
3. **Harvest** (in `step()`): when the agent enters a NEW frontier cell (or improves its score), snapshot
   `pyboy.save_state()` into the archive. Throttle to frontier cells (egg-in-hand / return_progress beyond the
   current max / in-gym) so cost stays bounded — Go-Explore's "remember promising states" step.
4. **Reset sampling** (in `reset()`): with prob `p_frontier` (e.g. 0.5, annealed down over training) load a
   cell sampled from the archive (weighted to HIGH-frontier, LOW-visit-count cells); otherwise load
   `start.state`. The `start.state` fraction keeps the run START-ANCHORED (the success gate is start-only), so
   the policy never drifts off the real task. KEY difference vs a fixed anchor: frontier states are EPHEMERAL,
   drawn from the policy ITSELF, and continuously refreshed — never a static foreign scene.
5. **The hard engineering problem — SubprocVecEnv has no shared memory.** The archive must be shared so a
   frontier discovered by one worker seeds resets in the others. Options: (a) **on-disk archive dir**
   (`runs/frontier_archive/<run>/cell_<hash>.state`, atomic writes, read at reset + in-proc LRU cache) —
   simple, robust, debuggable, survives restarts, load cost negligible vs a 65k-step episode; (b)
   `multiprocessing.Manager` dict — in-memory, faster, but slow under contention and not restart-safe; (c)
   `DummyVecEnv` single-process — trivially shared but loses 12-env throughput (PyBoy is CPU-bound).
   **Recommendation: (a).**
6. **Metrics/guardrails**: keep the origin split — `nav/*` = start envs (the GATE), add `front/*` = frontier
   envs. The transfer test the anchors failed: as frontier envs feed the SHARED value function, `nav/*` must
   IMPROVE (here it should, because the frontier states are start-continuous). ABORT if `front/*` moves while
   `nav/*` stays pinned (would mean even own-trajectory resets segregate — a deeper finding worth knowing).

**Phase plan:**
- Phase 1: cell key includes the egg flags → frontier cells = "egg in hand, undelivered, south of
  Cherrygrove" → thousands of short rollouts that START near the lab holding the egg → the Elm A-press is
  discovered in-distribution, then its value propagates up the start trajectory. Gate: `nav/egg_delivered_rate ≥ 0.8`.
- Phase 2: same machinery; with the story gate open the frontier extends through Route 31 → Violet → into the
  gym; cell key includes `gym_trainers_beaten` so "mid-Falkner-fight" becomes a sampled frontier → the fight is
  practiced in-distribution. Gate: `nav/badge_rate ≥ 0.5`.

**Build size / risks:** new module (archive + frontier sampler) + env `reset()/step()` hooks + train wiring.
Primary risk: the SubprocVecEnv sharing + save-state I/O cost (mitigated by on-disk archive + LRU cache + frontier
throttle). Secondary: archive growth/quality (bounded size + frontier-weighted eviction). Smoke path:
save/load round-trips a PyBoy state, archive populates from a random rollout, reset samples it, `nav/`+`front/`
split logs. **Estimated 20–30M-step validation run** with an early-abort gate (Phase-1 `nav/egg_delivered_rate`
rising off 0 by ~20M) before committing the full budget.

**DECISION REQUESTED:** approve building this frontier-reset architecture (Phase 1 first), or redirect. I am
pausing here rather than launching another pure-start run, per the escalation rule.

**HUMAN DECISION (2026-06-13): APPROVED — build the frontier-reset architecture, Phase 1 first.**

---

## Agent 059 — GO-EXPLORE / FRONTIER RESET (start-continuous), warm 053@80M (2026-06-13, LAUNCHED)

First implementation of the approved architecture. New module `env/frontier_archive.py` (TDD, 9 unit tests +
2 PyBoy integration tests green): an on-disk, process-shared cell→save-state archive. A fraction
(`FRONTIER_P`=0.5) of env resets restart from a state SAMPLED from the archive (cells harvested from the
policy's OWN trajectory) instead of `start.state`. The reset states are the policy's own visited pixels, never
a foreign save-state → no visual island → no segregation (the 047-057 failure mode is structurally impossible).
- **Mechanism**: `cell_key` = (map, coords//4, egg_received, egg_delivered, gym_trainers_beaten) — story flags
  SPLIT the cell so "Cherrygrove WITH egg" is distinct. `frontier_score` ranks pre-egg < egg-carrying (deepens
  with return_progress) < delivered/forward (deepens with waypoint+gym). Harvest fires only on frontier cells
  (egg in hand / in gym) → bounded archive, lazy 200KB save only on a new/improved cell. Sampling is ε-greedy
  over frontier score (10% uniform floor for diversity). Shared across SubprocVecEnv workers via the
  filesystem (atomic tmp+rename); cleared per run so no stale off-policy cells.
- **Why it should beat 053-058**: the start policy only reaches Cherrygrove-with-egg (return_progress 1-2),
  never the lab — so breadcrumbs/entropy never let it PRACTICE the backtrack or the Elm A-press. Here the
  archive bootstraps progressively: Cherrygrove-with-egg cells seed resets → the agent occasionally reaches
  Route29-with-egg → that cell is archived → resets sample it → … → lab-with-egg → A-press discovered by
  repetition. Each hop is a short, in-distribution practiced segment; the 600-step barren backtrack is never
  crossed in one shot. The +30 delivery reward then enters the SHARED value fn from states continuous with the
  start trajectory, so it transfers back to true-start episodes (the transfer the anchors couldn't make).
- **Single new variable = the architecture.** Reverted 058's entropy schedule (it de-locked pickup) to a FIXED
  0.01 (053's value) — the frontier reset supplies exploration via state diversity, not action noise. Kept the
  directional tile-reset + return breadcrumbs (harmless). Warm 053@80M (pickup 1.0). lr 8e-5, win cap 2, 65k.
- **Metrics**: `nav/*` = true start.state episodes (the SUCCESS GATE, uncontaminated); `front/*` = frontier
  episodes (begin deep along the trajectory). Frontier episodes carry `from_start=False`.
- **Boot confirmed**: frontier archive enabled, warm-start OK, ent_coef 0.01 constant, ~2900-3300 fps.
- **Early-abort gates**: ~5-8M `front/return_progress_mean` climbing past 3 (archive deepening south past
  Cherrygrove — the thing 058 never did) AND `nav/egg_received` HELD ≥ 0.6 (pickup not de-locked); ~15M
  `front/egg_delivered_rate` LEAVING 0 (A-press discovered at archived near-lab cells); ~30M
  `nav/egg_delivered_rate` LEAVING 0 (TRANSFER to true-start — Phase-1 milestone progress). ABORT if by ~15M
  `front/return_progress` stuck ≤ 2 (archive not deepening → mechanism failing) or `nav/egg_received` < 0.4.

**Run outcome (ABORTED at ~5M on the pickup-cratered gate): the MECHANISM WORKS, but p=0.5-on-all-envs
forgot the start pickup.** Two clear, separable signals:
- ✅ **Frontier deepens as designed**: archive grew 41→166 cells and scores climbed 101→**103** (egg-carrying
  at Route 29); `front/return_progress_mean` rose 1→2→**3** and held — frontier episodes reliably reach Route 29
  with the egg, the progressive bootstrap 058 never achieved. The on-disk shared archive + harvest + ε-greedy
  sampling all work in the live run.
- ❌ **Start pickup CRATERED**: `nav/egg_received` fell from ~1.0 (053 warm base) to a sustained **0** by ~3M,
  while `nav/reach_cherrygrove` stayed ~1.0 — the start envs reach Cherrygrove but no longer make the NORTHWARD
  detour to Mr.Pokemon's for the egg. `ep_rew_mean` drifted down 3.31→2.28.
- **Root cause (behavioral interference, NOT visual segregation)**: pickup is a NORTH excursion; every frontier
  episode (egg-in-hand at Route 29) practices going SOUTH. At p=0.5 across all 12 envs the southward gradient
  dominated the shared policy and suppressed the northward pickup, even though the egg flag is in the obs. The
  archive self-sustained from the frontier episodes' own progress, so it kept deepening WHILE the base task
  rotted — and pickup=0 structurally caps `nav/egg_delivered` at 0, so this is fatal, not waitable.
- **Lesson / fix**: the frontier reset must not starve the base-task gradient. → Agent 060 DEDICATES envs
  (9 pure start / 3 frontier) so the start→pickup→deliver task keeps full-strength reinforcement while a minority
  deepens the shared archive. **Kept** (works): the whole frontier-archive machinery + the warm-053 base + ent
  0.01. **Ruled out**: frontier reset at high p on ALL envs (forgets the base task via directional interference).

## Agent 060 — FRONTIER RESET with DEDICATED envs (9 start / 3 frontier), warm 053@80M (2026-06-13, LAUNCHED)

Single variable vs 059: the reset MIXING. 059's p=0.5-on-all-12 let the southward frontier gradient kill the
northward pickup (nav/egg_received 1.0→0). 060 dedicates the last 3 of 12 envs to frontier resets (p=1.0); the
other 9 are PURE start (p=0). The archive is still SHARED across all 12 (so the 3 frontier envs' deepening
practice enters the same value fn the 9 start envs train), but the start-task gradient is now 3× the frontier
gradient → pickup protected. Everything else identical to 059 (same archive code, harvest, ε-greedy sampling,
warm 053@80M, ent fixed 0.01, lr 8e-5, win cap 2, 65k).
- **Boot confirmed**: frontier_envs=3/12, warm-start OK, ent 0.01, no errors.
- **Gates**: ~8M `nav/egg_received` HELD ≥ 0.6 (the 059 failure must be fixed — THE primary check) AND
  `front/return_progress` ≥ 3 (frontier still deepening with fewer envs); ~20M `front/return_progress` → 4-5
  (reaching the lab) and `front/egg_delivered` LEAVING 0 (A-press); ~35M `nav/egg_delivered` LEAVING 0 (transfer).
  ABORT if `nav/egg_received` craters < 0.4 again (dedication insufficient → interference is fundamental, would
  need stronger egg-flag conditioning or frontier resets only from delivered/forward cells).

**MID-RUN @~14M — TWO firsts: pickup HELD + the DELIVERY A-PRESS DISCOVERED in-distribution.**
- ✅ **059 failure FIXED**: `nav/egg_received` holds ~0.7-1.0 (mostly 1.0, noisy) — dedicating 9 start envs
  protected the northward pickup. No sustained crater. The dedication hypothesis is confirmed.
- ✅✅ **Frontier reached the lab AND delivered**: archive depth climbed past 059's wall (103) to a full
  distribution — 8×score-103 (Route29), 45×104 (NewBark), **163×105 (egg-in-hand AT THE LAB)**, **58×200
  (DELIVERED — ELM_BIT set)**, **26×210 (delivered + past the gate)**. The frontier bootstrap walked the
  backtrack from progressively deeper footholds and the agent PRESSED A AT ELM — the delivery action never
  performed in 350M+ cumulative steps across all prior runs. `front/return_progress` now hits 4-5 regularly.
- The 200/210 cells mean delivered frontier episodes are already advancing PAST the open story gate (Phase-2
  frontier emerging via the same score-weighted auto-curriculum, no new code).
- **STILL OPEN — the transfer**: `nav/egg_delivered` (true start episodes) = 0 at 14M. Delivery is practiced
  from frontier footholds and its +30 value is now in the SHARED value fn; whether the 9 start envs learn to
  carry the egg through to Elm from start.state is the ~35M gate. Letting it run — no change (working as designed).
- **@~19M update**: pickup still held (`nav/egg_received` ~0.6-0.7), `front/egg_delivered` now FIRING (0.5-1.0 —
  frontier delivery consolidated), archive advancing into Phase 2 (186×score-220 = delivered + Route-30 gate).
  BUT `nav/egg_delivered` STILL 0 — no transfer yet. **WATCH-ITEM (prime suspect if transfer fails at ~35M)**:
  `frontier_score` for delivered cells is UNBOUNDED (200+10·waypoint) while carry cells cap at 105, so
  score-weighted sampling increasingly favors Phase-2 cells and STARVES the delivery-backtrack practice before
  Phase-1 transfers. Candidate fix for agent_061 if needed: cap/separate sampling so carry-state (≤105) cells
  keep a reserved share until `nav/egg_delivered` lifts. NOT changing mid-run (unconfirmed; let the gate decide).

**Run outcome (ABORTED at ~24M — delivery DISCOVERED but never TRANSFERRED to start; the watch-item was the
cause).** The headline: the frontier reset achieved the project-first that 350M+ steps never did — the egg
delivery A-press, practiced and consolidated in frontier episodes (`front/egg_delivered` ~0.8). But it did NOT
transfer to true-start episodes:
- `nav/egg_delivered` = **0 across the ENTIRE run** (0 deliveries ever); `nav/return_progress` dead-flat
  (windowed quartiles 0.91/1.10/1.00/0.95 — start episodes sit at Cherrygrove-with-egg, never push south);
  `nav/egg_received` held ~0.65 (pickup fine); `ep_rew_mean` flat ~2.4. The archive FROZE by 19M (105-cells
  163→163, 220-cells 186→188 over 5M) — the system equilibrated.
- **Root cause (confirmed the watch-item, refined)**: `frontier_score = 100 + episode-max-return_progress`.
  Because the score is the EPISODE max and `add()` replaces a cell when a higher score arrives, a DEEP episode
  passing through a physically-SHALLOW carry cell overwrote that cell's score to a deep value. Result: 163 cells
  at the lab (105) but **~0 at Cherrygrove (102)** — the shallow carry states WHERE THE START EPISODES ARE were
  destroyed. Score-weighted sampling then almost never reset there, so "push south from Cherrygrove with the
  egg" was never practiced → no transfer. (Not Phase-2 dominance per se; it's depth-bias erasing shallow cells.)
- **Decision**: aborted at 24M rather than waiting the ~35M gate — the archive was frozen and trends dead-flat
  (a known wall; the brief forbids burning GPU to re-confirm it), and the fix is independently better.
- **Kept** (works, huge): the dedicated-env design (pickup held), the whole archive/harvest machinery, and the
  proof that frontier reset DISCOVERS the delivery in-distribution. **Ruled out**: depth-based `frontier_score`
  (episode-max-return_progress) — erases shallow carry cells, blocks the transfer.

## Agent 061 — FLAT-TIER frontier score (fix the transfer), warm 053@80M (2026-06-13, LAUNCHED)

Single variable vs 060: `frontier_score` only. Now a FLAT priority tier from the cell's OWN flags — carry 2 >
delivered 1 > pre-egg 0 — with NO depth term. Two effects: (1) a cell's score never changes, so `add()` keeps
the FIRST captured state per cell → shallow Cherrygrove-with-egg cells PERSIST (no longer overwritten to deep
scores); (2) all carry cells are equal, so ε-greedy samples them UNIFORMLY across depth → the shallow carry
states where start episodes sit get practiced as much as the lab. carry > delivered keeps Phase-1 transfer from
being starved by Phase-2 cells. Everything else identical to 060 (dedicated 9 start / 3 frontier, warm 053@80M,
ent 0.01, lr 8e-5, win cap 2, 65k; fresh per-run archive). TDD: score test rewritten to assert flat tiers
(RED on 060's depth-score, GREEN after) + 2 PyBoy integration tests + base smoke all green. Boot confirmed.
- **Gates**: ~10M archive shows a SPREAD of carry depths incl. shallow (score-2 carry cells at Cherrygrove/
  Route29/lab, not just lab) AND `nav/egg_received` HELD ≥ 0.6; ~25M `nav/return_progress` climbing past 2
  (start episodes carrying the egg SOUTH — the thing 060 never did) and `nav/egg_delivered` LEAVING 0 (the
  transfer); ~40M `nav/egg_delivered` ≥ 0.5 toward the Phase-1 milestone (≥ 0.8). ABORT if by ~25M
  `nav/return_progress` still flat ~1 (flat tiers insufficient → the transfer barrier is deeper than sampling).

**Run outcome (ABORTED ~17M — archive composition FIXED, but transfer STILL fails; start task ERODES).**
The flat-tier fix worked at the level it targeted but did not move the gate:
- ✅ Archive composition fixed: carry cells now span the WHOLE corridor incl. shallow Cherrygrove (26_3) — the
  thing 060 erased. Frontier delivers reliably (`front/egg_delivered` fired in 24 rollouts).
- ❌ No transfer: `nav/egg_delivered` = 0 throughout; `nav/return_progress` quintiles DECLINED 1.21→0.51 (start
  episodes get LESS deep over time, not more); `nav/egg_received` eroded 0.58→0.48 (053 base was 1.0).
- **Stochastic eval (061@15M, 20 eps from start.state, 12k-cap)**: badge 0/20, **egg received 1/20, delivered
  0/20**, cherrygrove 17/20, route31 0/20. Battles W/F/L **132/410/2** (~20 wild flees/ep, in_battle 20%). The
  start policy reaches Cherrygrove then WANDERS/flees — it doesn't reliably make the northward Mr.Pokemon's
  detour to even pick up the egg, let alone carry it south to deliver.

## ESCALATION (2026-06-13/14): the FRONTIER TRANSFER WALL — design proposal, PAUSE for human review

> Across 3 frontier-reset runs (059/060/061) the transfer metric `nav/egg_delivered` has NOT moved off 0.
> Per the operating-protocol escalation rule (metric unmoved after several runs; only architectural
> hypotheses remain), STOP autonomously tweaking and write this up. No agent_062 launched.

**What is PROVEN (keep):** the Go-Explore frontier reset RELIABLY DISCOVERS + practices the egg-delivery
A-press in-distribution (`front/egg_delivered` ~0.8) — the action never performed in 350M+ prior steps, the
project's hardest-ever sub-skill. The on-disk shared archive, harvest, ε-greedy sampling, dedicated-env
machinery, and flat-tier scoring all work as designed. This is a genuine breakthrough.

**The WALL:** the frontier-practiced delivery does NOT transfer to the from-start policy, and the frontier
reset actively DEGRADES the base task. Consistent across all 3 runs:
- 059 (p=0.5 all envs): southward gradient killed northward pickup (nav/egg_received 1.0→0).
- 060 (dedicated 9/3, depth-score): delivery discovered, archive frozen, nav/egg_delivered flat 0.
- 061 (flat-tier): archive fixed, frontier delivers, but nav pickup eroded 1.0→0.5 and return_progress declined.
- ROOT: the 3 always-post-pickup frontier envs shift the SHARED policy's mass toward carry/delivery behaviors
  and AWAY from the pre-pickup northward approach; 9 start envs + warm-053 don't hold it. There is a structural
  tension — practicing delivery requires resetting post-pickup, which erodes the pickup the start task needs.

**Candidate directions (ranked; need a human steer — some are architectural / have real trade-offs):**
1. **Concentrate frontier gradient on the backtrack** — TERMINATE frontier episodes shortly after delivery
   (or shorten their MAX_STEPS) so their 65k steps aren't dominated by off-task post-delivery Phase-2 wandering
   that erodes the start policy. Cheapest, directly targets the dilution; env-only change. (My recommended first.)
2. **Two-stage / annealed frontier intensity** — re-consolidate pickup first (0 frontier envs ≈ reproduce 053),
   THEN ramp frontier in once pickup is solid, so delivery is added ON TOP of a stable pickup rather than
   competing with it from step 1. More moving parts.
3. **Fewer frontier envs (2 or 1 of 12)** — accept slower frontier to protect pickup further (061's 3/12 still
   eroded it). Cheap but may just slow the erosion, not stop it.
4. **Representational** — strengthen the egg-flag salience so the policy can do opposite N/S navigation at the
   same Cherrygrove tile (e.g., blend a frontier/egg-state channel into the RGB, PWhiddy v1 style). Architectural,
   higher cost; the obs vector's single egg bit may be too weak for the CNN to switch direction on.
5. **Bank the win and PIVOT the framing** — the frontier PROVES delivery is reachable and practiceable; consider
   a different transfer mechanism (e.g., the frontier policy as a teacher / BC-style distillation into the start
   policy), or accept Phase-1-via-frontier as "solved in frontier episodes" and re-scope.

**Also seen (lower priority):** the ~20-flee/ep wild-battle tax wastes episode time; a flee/time pressure could
help but battle penalties historically taught grass-avoidance (ruled out) — revisit only if it gates progress.

**DECISION REQUESTED:** which direction (1-5, or combine)? I recommend starting with (1) — cheapest, directly
addresses the diagnosed dilution/erosion — with a 20-30M early-abort validation. Paused pending your call.

**HUMAN DECISION (2026-06-14): delegated — "do what you judge best and track it in the log." → option (1).**

## Agent 062 — TERMINATE frontier episodes after delivery (stop the erosion), warm 053@80M (2026-06-14, LAUNCHED)

Single variable vs 061: frontier episodes now END the moment they deliver the egg (and truncate at
`FRONTIER_MAX_STEPS`=8000 if they don't). Rationale (refined from the 061 eval): a 65k-step frontier episode
delivers in ~1-2k steps then spends ~63k steps in OFF-TASK post-delivery wandering (Phase-2 grass, fleeing);
that aimless gradient — not delivery practice — is what eroded the start policy into wandering (eval: egg 1/20,
20 flees/ep). Ending on delivery (a) removes the wandering-erosion gradient and (b) turns each frontier env into
a stream of short, clean carry→deliver episodes (concentrated practice + more frequent archive resampling).
Code: `truncated` includes `(self._from_frontier and just_delivered)` and the per-episode cap is
`frontier_max_steps` for frontier episodes vs full `MAX_STEPS` for start episodes (env). Everything else
identical to 061 (dedicated 9/3, flat-tier score, warm 053@80M, ent 0.01, lr 8e-5, win cap 2). TDD: new
`test_frontier_episode_uses_short_cap` + all prior archive/env/base smoke tests green. Boot confirmed.
- **Gates**: THE primary check — `nav/egg_received` must STOP eroding and recover toward ≥ 0.6 (061's erosion to
  0.5 must reverse; if frontier wandering was the cause, removing it should let the 9 start envs hold pickup);
  ~15M `front/egg_delivered` still firing (short episodes still deliver) AND `front` ep_len_mean ≈ small (cutoff
  working); ~25-30M `nav/return_progress` climbing past 2 and `nav/egg_delivered` LEAVING 0 (the transfer).
  ABORT if by ~20M `nav/egg_received` still ≤ 0.5 (the erosion wasn't from post-delivery wandering → escalate to
  option 2/4).

**@~27M — THE FIX WORKS: erosion stopped AND transfer STARTED (first healthy start delivery).** The two skills
that destroyed each other in 059-061 are now BOTH improving:
- `nav/egg_received` recovered + climbing: 5ths 0.53→0.64→0.66→0.66→**0.72** (abort-gate passed; vs 061's decline).
- `nav/return_progress` climbing MONOTONICALLY: 5ths 0.61→0.75→0.94→1.11→**1.21** (vs 061's DECLINE 1.21→0.51).
- ✅ **`nav/egg_delivered` LEFT ZERO** — a true start episode delivered (1 rollout). Unlike 054's lone fluke (on
  a DEGRADING base), this rides a healthy climbing context → looks like the leading edge of real transfer.
- `front` delivers robustly (34 rollouts; 212 delivered cells archived) — strong delivery value in the shared fn.
- Diagnosis CONFIRMED: the 059-061 erosion was the off-task post-delivery wandering gradient; terminating
  frontier episodes on delivery removed it, letting pickup heal AND the delivery value transfer to start.
- NOT claiming Phase-1 done yet — needs `nav/return_progress` past 2-5 and `nav/egg_delivered_rate` rising
  toward the milestone (≥0.8 in stochastic eval). Letting it consolidate (~50-80M).

**@~35M — the 27M delivery did NOT consolidate (one-off, like 054).** `nav/egg_delivered` still just 1 rollout
total (6ths 0/0/0/0/0.013/0); `nav/return_progress` climbed to 1.13 then REVERTED to 0.68 (last window);
pickup plateaued ~0.6. 062 is the BEST frontier run (erosion fixed, return_progress reached ~1.1 vs 061's
decline) but the transfer is PLATEAUING at Cherrygrove-with-egg (~return_progress 1) — start episodes pick up
the egg but don't push south to the lab (the agent-054 barren-gap wall, at a higher floor). Decision criterion
@~42M: delivery rising + return_progress climbing past ~1.5 → consolidating (continue); else plateaued → this is
the 4th frontier run without consolidated transfer → escalate to human with the full picture + next candidate
lever (more frontier envs, now safe since terminate-on-delivery removed the wandering that made high p erode).

**Run outcome (ABORTED ~42M — PLATEAUED then REGRESSED; transfer never consolidated).** Recent-third means:
`nav/egg_delivered` 0.000 (1 delivery in 562 rollouts total — the 27M one-off, never repeated),
`nav/return_progress` 0.61 (down from 1.13 peak), `nav/egg_received` **0.40** (pickup eroding AGAIN, down from
0.72). The terminate-on-delivery fix DELAYED the erosion (062 reached return_progress 1.1 + 1 delivery, the best
of any frontier run) but over 42M the frontier gradient still pulled the shared policy off pickup.
- **CONSOLIDATED CONCLUSION across 059-062 (4 frontier runs)**: the Go-Explore frontier reset reliably DISCOVERS
  + practices the egg-delivery A-press (a genuine project-first), but its delivery skill does NOT transfer to the
  from-start policy, because there is a FUNDAMENTAL tension — transfer pressure ⟺ pickup erosion — that no
  mixing ratio / sampling / episode-length tweak escapes. ROOT (representational): at Cherrygrove the obs IMAGE
  is identical with/without the egg, so the CNN (which dominates the policy) cannot separate "go N to pick up"
  from "go S to deliver"; the egg bit in the obs VECTOR is too weak to override the image. → the next fix must
  be representational, not another mixing tweak.

## Agent 063 — EGG-STATE VISUAL MARKER (representational fix), warm 053@80M (2026-06-14, autonomous per delegation)

Single variable vs 062: the obs IMAGE now carries an 8x8 top-left corner patch encoding the egg quest state —
carrying=red (255,0,0), delivered=green (0,255,0), none=black (0,0,0) — stamped in `_get_obs`. This gives the
CNN a trivially-detectable visual signal to SEPARATE pre-pickup navigation (go N) from carrying navigation
(go S to deliver) at the SAME Cherrygrove tile — the root cause of the 059-062 transfer⟺erosion wall. The
image shape stays (72,80,3) so 053's checkpoint warm-loads unchanged (the policy learns to attend to the patch).
Everything else identical to 062 (frontier 9/3 dedicated, flat-tier, terminate-on-delivery, ent 0.01, lr 8e-5,
win cap 2). TDD: new `test_egg_state_marker_in_obs` (no-egg=black, carrying=red) + all prior tests green. Boot OK.
- **Bet**: with the egg state VISIBLE to the CNN, the carrying-policy and pickup-policy stop overwriting each
  other → pickup holds AND the frontier-practiced delivery (keyed to the red marker) transfers to start episodes.
- **Gates**: ~15M `nav/egg_received` HELD ≥ 0.6 AND NOT eroding late (the 062 erosion must be gone); ~25M
  `nav/return_progress` climbing past 2 (start episodes carrying south); ~35-50M `nav/egg_delivered_rate` rising
  off 0 and CONSOLIDATING (multiple rollouts, not a 1-off) toward the Phase-1 milestone (≥ 0.8 in eval).
- **FIRM ESCALATION**: this is the principled representational fix; if it ALSO fails to consolidate transfer
  (pickup erodes again OR delivery stays a one-off by ~40M), STOP — do NOT launch a 6th frontier run; escalate
  the bigger architectural / pivot decision (e.g. policy distillation) to Fabio.

(Decision rationale: Fabio delegated — "do what you judge best, track it in the log." 4 frontier runs put us at
the escalation threshold, but the diagnosis pointed cleanly to ONE representational fix that is cheap and
warm-compatible, and the GPU was idle — so I ran it rather than idle for hours, with the firm escalation above.)

**Run outcome (ABORTED ~23M on the hard criterion — WARM marker COLLAPSED pickup; firm escalation triggered).**
`nav/egg_received` recent-third = **0.04** (raw last 25 almost all 0) — far below the 0.4 abort line, and WORSE
than 062's 0.4. The warm-053 CNN was corrupted by the new corner patch rather than helped by it (it never
learned to attend to the marker; pickup just collapsed). `nav/return_progress` 0.10, `nav/egg_delivered` ~0.01
(5/274). The warm start CONFOUNDS the marker hypothesis (can't conclude the representational idea is wrong —
only that warm-loading it into a marker-naive policy fails). Per the firm pre-registration: STOP, no 6th
frontier run, escalate.

## ESCALATION #2 (2026-06-14): frontier-reset wall after 5 runs — strategic fork, PAUSE for human

> 059-063: five runs. The Go-Explore frontier reset RELIABLY DISCOVERS + practices the egg-delivery A-press
> (front/egg_delivered ~0.8-1.0 every run — the project's hardest sub-skill, a genuine first). But it CANNOT
> compose that with the pickup skill in one shared RL policy from start.state: every run hits the transfer⟺
> erosion tension. Mixing ratio (059), depth-score (060), flat-tier (061), terminate-on-delivery (062), and the
> warm egg-marker (063) each failed to consolidate `nav/egg_delivered`. The root is representational/structural:
> the shared CNN can't stably hold "no egg → go N to pick up" AND "carrying → go S to deliver", and a warm obs
> patch can't fix it. This is no longer a knob — it needs a strategic decision.

**Strategic options (need human steer; budget/architecture implications):**
A. **COLD-START with the egg-marker** — train from scratch WITH the marker so the CNN learns to use it cleanly
   (no warm corruption). The cleanest test of the sound representational idea; if it works it could unlock
   pickup+delivery coexistence. COST: ~tens of M to relearn navigation from zero (hours of GPU), and still
   unproven the marker enables separation. (Recommended IF we want to give the representational fix a fair shot.)
B. **DISTILLATION / teacher-student** — we already HAVE two competent specialists: 053 (pickup-from-start) and
   any frontier checkpoint (delivery-from-carry). Train one policy by behavioral cloning to imitate pickup when
   no-egg and delivery when carrying (egg state as input) → explicitly COMPOSE the two skills instead of hoping
   shared RL balances them. Most directly targets the actual failure; a new (BC/DAgger) build.
C. **RE-SCOPE Phase 1** — bank "frontier discovers delivery" as the result; accept delivery-from-frontier and
   either move to Phase 2 from frontier states, or declare the from-start delivery a known hard limit and
   redefine the milestone. Cheapest; concedes the strict start-state gate.
D. **Different architecture entirely** (e.g. recurrent policy, or separate pickup/delivery heads gated by the
   egg flag) — bigger redesign.

**RECOMMENDATION:** B (distillation) is the most direct fix for the diagnosed problem (composing two known-good
skills); A (cold-marker) is the cleanest test of the representational hypothesis if we prefer to stay in pure
RL. Both are real builds — hence pausing for the decision rather than autonomously committing GPU-hours.
**No run launched; GPU idle pending the call.**

## Agent 064 — DISTILLATION step 1 (gated mixture) → PREMISE INVALIDATED, REFRAME (2026-06-14)

Fabio chose distillation. Before a BC pipeline I ran the cheap de-risk: a hand-gated MIXTURE-OF-EXPERTS
(`agents/rl/eval_gated_mixture.py`) — 053 (pickup) while no egg, switch to 062 (delivery) on the egg RAM bit.
Also gated the failed egg-marker OFF by default (`egg_marker` param, so teachers see the obs they trained on).
**Result — composition FAILS, and it exposed a misdiagnosis we'd carried for 5 runs:**
- Gated 053+062 from start.state: egg received 12/20, **delivered 0/20** — every carrying episode stalled at
  route30_gate (the pickup area), never starting the southward backtrack.
- 062 ALONE delivery skill, mapped by start state: `before_elm` (AT lab) **6/12 (~50%, wanders N the rest)**;
  `newbark_egg` (adjacent to lab) **0/12 (wanders N to Cherrygrove 10/12)**; `mid_route30` (Cherrygrove) **0/12**.
- **REFRAME (the real wall, now precise)**: there is NO delivery-NAVIGATION skill in any policy. 062's only
  delivery competence is a ~50% A-press WHEN PLACED ON ELM'S DOORMAT. "front/egg_delivered ~0.8" across 060-062
  was an ARTIFACT of frontier resets landing at/near the lab — it never reflected the agent NAVIGATING the
  carry-backtrack. The unsolved skill is **carrying the egg SOUTH (pickup → Cherrygrove → Route29 → NewBark →
  Elm)**, which fights a strong NORTHWARD bias every agent has (all tile-exploration income was northward;
  carrying south crosses already-visited, zero-income tiles toward a distant +30). This is the agent-054
  barren-gap wall, now definitively localized.
- **Consequence**: distillation of 053+062 is NOT viable — no carry-navigation teacher exists to clone (you
  can't distill a skill no policy has). The composition test did its job: it cheaply invalidated the premise.
- **Kept (reusable)**: `eval_gated_mixture.py`, the `egg_marker` config gate (marker default OFF), all the
  frontier-archive machinery.

**REVISED OPTIONS (the real problem = learn carry-navigation SOUTH against the northward bias):**
1. **Strong southward-carry shaping** — scale the return-leg breadcrumbs WAY up (currently 1-3) into a dense
   southward potential while carrying the egg, so going south PAYS enough to beat the northward exploration
   bias. Most direct attack on the diagnosed bias; reward-only change. Risk: shaping history is mixed (can hack/
   not generalize), but it has NOT been tried at strong magnitude with the current clean setup.
2. **Build a carry-navigation specialist, THEN distill** — a dedicated run that learns deliver-from-carry from
   the carry save-states (before_elm/newbark/mid_route30); but training from fixed carry states is the
   ruled-out ANCHOR/segregation pattern → likely needs the frontier reset done RIGHT (no terminate-on-delivery;
   reward full end-to-end southward carry; ensure the whole Cherrygrove→lab path is consolidated, not just
   near-lab resets). I.e. a corrected frontier run aimed at carry-NAV, not the A-press.
3. **Re-scope** — accept that pure from-start delivery is the project's fundamental wall (10+ agents), bank the
   diagnosis, and either pursue Phase 2 from delivered states or redefine the milestone.
RECOMMENDATION: (1) is the cheapest direct test of the bias hypothesis and not yet tried at strength; if it
moves nav/return_progress past Cherrygrove it validates the path. PAUSED for Fabio's steer (premise changed).

**HUMAN DECISION (2026-06-14): option (1) — strong southward-carry shaping. → agent_064 LAUNCHED.**
Single variable vs 062 (rewards.py only): `RETURN_BREADCRUMBS` scaled from 1/2/3/3/2 to a steep increasing
southward staircase **2/5/12/20/30** (sum 69 pre-scale ≈ 7× the northward exploration income), with a big jump
at Route 29 (the diagnosed Cherrygrove→Route29 barren-gap stall). Positive-only + latched (no pickup-dodge, no
hacking). Kept ALL of 062's config (frontier 9/3 dedicated, flat-tier, terminate-on-delivery, ent 0.01, lr
8e-5, win cap 2, egg_marker OFF) — the frontier resets put the agent in the carry states so the strong shaping
is actually EXPERIENCED (pure-start 053 never reaches them). Warm 053@80M. Smoke tests green, boot OK.
- **Bet**: making the southward carry the most profitable thing on the map overcomes the northward bias → the
  start policy learns to carry the egg south past Cherrygrove (the thing NO policy has done) → `nav/return_progress`
  climbs past 2-3 toward the lab and delivery consolidates.
- **Gates**: ~15M `nav/return_progress` climbing past 2 (start episodes crossing Cherrygrove→Route29 southward —
  the carry-nav the diagnosis says is missing) WITH `nav/egg_received` held ≥ 0.5; ~30M `nav/return_progress`
  past 3-4 and `nav/egg_delivered_rate` rising off 0 and CONSOLIDATING; ~50M toward the Phase-1 milestone (≥0.8).
  ABORT if by ~20M `nav/return_progress` still stuck ≤ 1.5 (strong shaping doesn't beat the bias → the wall is
  deeper than reward magnitude; escalate to option 2 corrected-frontier or re-scope).

**Run outcome (ABORTED ~23M — strong sparse shaping FAILED and DESTABILIZED).** Recent-third: `nav/return_progress`
carry-depth-when-holding = **1.00** (start episodes NEVER cross south); `front/return_progress` declined
monotonically 2.70→2.32→1.91→**1.54** (frontier episodes REGRESSING on carry-nav); `nav/egg_received` eroded
0.52→0.33; `nav/egg_delivered` 0/300. Two compounding faults: (1) breadcrumbs are SPARSE — the +12 sits BEYOND
the barren gap, no gradient ACROSS it; (2) the large rewards (up to +30, norm_reward=False) destabilized the
value fn (correlates with the front-decline + pickup-erosion). **Ruled out: scaling SPARSE map-entry breadcrumbs
— adds instability without adding a gradient; doesn't cross the barren gap.**

## ESCALATION #3 (2026-06-14): the CARRY-NAVIGATION wall is fundamental — recommend re-scope / rethink

> Session arc: 5 frontier runs (059-063) → gated-mixture diagnosis (064) reframed the problem → strong sparse
> shaping (064) failed+destabilized. The from-start egg-carry-SOUTH (pickup→Cherrygrove→Route29→NewBark→Elm)
> remains UNSOLVED, as it has for ~12 agents across the whole project.
>
> **Deepest diagnosis yet**: the northward bias is NOT an active reward (carrying north re-treads known tiles,
> ~0 income) — it is an ENTRENCHED POLICY HABIT in the warm weights. The start policy learned northward
> navigation (start→pickup) and applies it even when carrying. Reward shaping can't overcome a habit the policy
> won't EXPLORE away from: sparse breadcrumbs are never reached; large rewards destabilize; and a dense
> potential (10f) caused the agent to dodge the carrying state. This is fundamentally an EXPLORATION/HABIT
> problem, and the frontier reset (the one tool that forces southward exploration by placing the agent there)
> also failed to consolidate it (059-063) — the practice didn't stick against the habit + the near-lab artifact.

**Options (honest assessment — none high-confidence; the wall is deep):**
1. **Dense small per-tile southward gradient** (NOT large/sparse) — e.g. +0.05 per new tile visited while
   carrying in the southern corridor (positive-only, latched per-tile, small so it doesn't destabilize like
   064's +30 spikes). Creates a continuous southward pull across the gap without value-fn instability. Cheapest
   remaining shaping idea; still uncertain (habit may resist). 
2. **Corrected frontier for carry-nav** — frontier reset, NO terminate-on-delivery, dense southward reward,
   reset DISTRIBUTION biased to the SHALLOW carry edge (Cherrygrove-with-egg, where the start policy actually
   is) so it practices the exact gap crossing repeatedly. A focused retry of the one tool that forces southward
   exploration. Multi-run.
3. **RE-SCOPE (recommended)** — the strict from-start-delivery gate has resisted ~12 agents + a thorough
   session. Options: (a) accept frontier/curriculum delivery and pursue PHASE 2 (Falkner) from delivered states
   to make progress on the OTHER wall; (b) redefine the milestone (e.g. delivery-from-Cherrygrove rather than
   pure start); (c) treat the carry-nav exploration problem as a research item needing a different method
   (intrinsic curiosity toward the lab, goal-conditioned RL, or a longer dedicated exploration budget).
RECOMMENDATION: (3) — we've thoroughly characterized this wall; continuing to throw reward tweaks at it is the
"motion is not progress" trap. Better to bank the (substantial) diagnostic progress and either pivot to Phase 2
or pick a fundamentally different method with Fabio. PAUSED; GPU idle.

**HUMAN DECISION (2026-06-14): option (1) — dense small per-tile southward gradient. → agent_065 LAUNCHED.**
Single var vs 062 (rewards.py): +0.05 per NEW (episode) tile while CARRYING the undelivered egg in the southern
corridor {Cherrygrove, Route29, NewBark, Elm}. Fixes BOTH 064 faults: DENSE (exists across the barren gap, not
a sparse bonus beyond it) and SMALL (no +30 value-fn destabilization). Mechanism: while carrying, north =
already-visited tiles (+0), south = new corridor tiles (+0.07 = 0.02 base + 0.05 carry) → a clear dense
southward gradient for PPO to shift the entrenched northward habit. Positive-only + episode-latched (no dodge/
hack/milk). Breadcrumbs reverted to 1/2/3/3/2; kept 062 config (frontier 9/3, flat-tier, terminate-on-delivery,
ent 0.01, warm 053). Boot OK.
- **Gates**: ~15M `nav/return_progress` carry-depth climbing past ~1.5 (start episodes crossing Cherrygrove→
  Route29 — the barren gap) AND `front/return_progress` climbing (not declining like 064) with pickup ≥ 0.5;
  ~30M `nav/egg_delivered` rising off 0 + consolidating. ABORT if ~20M carry-depth still ~1.0 (the dense
  gradient also can't shift the habit → the wall is exploration-fundamental; re-scope per ESCALATION #3 option 3).

**Run outcome (ABORTED ~22M — CARRY-NAV WALL CRACKED, but pickup collapsed → the tension is REPRESENTATIONAL,
definitively).** The most informative run of the session:
- ✅✅ **Carry-navigation SOLVED (a project first, ~12 agents stuck)**: `nav` carry-depth-when-holding reached
  **4.00** — start episodes that hold the egg now carry it to NEW BARK (adjacent to Elm). `front/return_progress`
  climbed 1.3→3.3 (vs 064's DECLINE). The dense small southward gradient shifted the entrenched northward habit
  where sparse/strong shaping couldn't. The carry-nav is LEARNABLE.
- ❌ **Pickup COLLAPSED to 0.02** (`nav/egg_received` 0.65→0.07→0.02). The dense southward reward shifted the
  policy south so hard it ERASED the northward pickup detour. 0 deliveries (start episodes no longer GET the egg).
- **DEFINITIVE DIAGNOSIS (proven from BOTH directions now)**: pickup (go N) and carry-nav (go S) CANNOT coexist
  in one CNN policy because the obs IMAGE is identical pre/post-pickup — push carry-nav up and pickup collapses
  (065), keep pickup and carry-nav never develops (053-064). This is exactly the "problema rappresentazionale"
  the Filone intro suspected — now PROVEN for the pickup⟺carry split. Reward shaping alone can NEVER solve it;
  the policy needs to SEE which mode it is in.
- **THE PATH IS NOW CLEAR — combine the two PROVEN pieces**: 065's dense southward reward (teaches carry-nav) +
  a WORKING egg-state visual marker (063's idea — lets the CNN separate the N-pickup mode from the S-carry mode
  so BOTH coexist). 063's marker failed ONLY because warm-loading it into a marker-naive 053 corrupted pickup;
  COLD (learn both modes from scratch WITH the marker present) avoids that. agent_066 candidate: COLD start +
  egg_marker=True + dense southward reward (+breadcrumbs + frontier reset). Cost: cold = ~tens of M (hours), but
  now strongly motivated (every piece validated). Early-abort if neither pickup nor carry-nav develops by ~15M.
- **Kept**: 065's dense reward (in rewards.py), the egg_marker gate (env), 065 checkpoints (carry-nav-competent).

## Agent 066 — COLD SYNTHESIS: egg-marker + dense southward reward (2026-06-14, Fabio: "you decide & proceed")

THE synthesis of the session — combines the two pieces 059-065 validated: 065's dense southward reward (teaches
carry-nav) + a WORKING egg-state visual marker (lets the CNN SEE the pickup vs carry mode so both coexist).
- **Config**: COLD (INIT_FROM_CHECKPOINT=None) + `EGG_MARKER`=True (corner patch carrying=red/delivered=green/
  none=black) + dense +0.05/new-carry-tile in the southern corridor + breadcrumbs (1/2/3/3/2) + frontier 9/3
  flat-tier terminate-on-delivery + win cap 2. lr **1.5e-4** (cold needs it; 8e-5 is a fine-tune value), ent 0.01.
- **Why COLD (not warm)**: 063 proved warm-loading the marker into a marker-naive policy corrupts pickup; 065's
  policy has a strong southward HABIT (warm from it would fight a northward-pickup re-learn). Cold = no
  corruption, no entrenched habit either way — the CNN learns pickup (no-marker→go N) and carry-nav (red→go S)
  as SEPARATE modes from gradient step 1. The principled clean test the warm runs couldn't be.
- **The bet (the whole session rides on this)**: with the egg state VISIBLE, pickup and carry-nav no longer
  fight (the 065 tradeoff dissolves) → the cold policy learns BOTH → delivers from start → first real progress
  to the Phase-1 milestone.
- **Cost**: cold = ~tens of M to relearn navigation (hours). Boot OK ("Training from scratch").
- **Gates (cold ⇒ patient)**: ~20M pickup developing (`nav/egg_received` ≥ 0.2, rising — learning start→pickup);
  ~40M pickup ≥ 0.5 AND carry-nav present (`nav/return_progress` > `nav/egg_received`, i.e. carry-depth > 1) —
  THE coexistence test the marker is meant to enable; ~60-80M `nav/egg_delivered` rising off 0. ABORT if by ~25M
  pickup still < 0.1 (cold not even learning navigation → setup problem) or if carry-nav develops but pickup
  collapses again like 065 (marker insufficient → escalate, likely re-scope).

**Run outcome (ABORTED ~26M — PURE-START COLD WALL, the agent_019 failure).** `nav/reach_cherrygrove` ~0.02
(never learned directed navigation to waypoint 1), `nav/egg_received` 0, `ep_rew` plateaued ~1.47 (positive,
so not 019's death-spiral, but stuck on navigation). Frontier archive empty (cold policy never reached carry
states → nothing harvested → the diversity mechanism stayed inert: chicken-and-egg). **Lesson: pure-start cold
can't bootstrap navigation without state diversity (agent_019); the marker/dense-reward synthesis needs the
policy to HAVE navigation first.** → agent_067 seeds the diversity.

## Agent 067 — SEEDED cold synthesis (egg-marker + dense reward + frontier seeded from 065), 2026-06-14

Fix for 066's pure-start-cold wall: SEED the frontier archive at launch with 065's 400 carry/delivered
save-states (`FRONTIER_SEED_FROM`, new train_cnn logic) so the 6 frontier envs (bumped 3→6) reset into the
delivery corridor FROM STEP 1 — supplying the state diversity that pure-start cold lacks. Because the seeded
carry states are in the SAME maps the pickup route crosses (Cherrygrove/Route29/...), the shared map-navigation
the frontier envs learn should bootstrap the 6 start envs toward pickup too. The marker (applied at obs-time, so
065's marker-less save-states are compatible) keeps the heavier 6-env southward practice from collapsing pickup
(the 065 failure). Else = 066: COLD, EGG_MARKER, dense +0.05/carry-tile, breadcrumbs, lr 1.5e-4, ent 0.01.
- **Boot**: SEEDED 400 cells, frontier 6/12, cold ("from scratch"), no errors.
- **Gates**: ~15M `nav/reach_cherrygrove` climbing (>0.2 — seeded diversity bootstrapping navigation, unlike 066's
  flat ~0.02) AND `front/return_progress` developing (frontier envs learning carry-nav from seeds); ~30M
  `nav/egg_received` rising (pickup developing) AND carry-nav present; ~50M+ the COEXISTENCE test (pickup ≥0.5 AND
  carry-depth >1, both held — the marker working) → `nav/egg_delivered` off 0. ABORT ~20M if reach_cherrygrove
  still <0.2 (seeding didn't bootstrap nav either → cold is the wrong vehicle; escalate to re-scope / warm tack).

**Run outcome (ABORTED ~24M — SEEDING WORKS for cold nav-bootstrap, but 6/6 carry-dominance COLLAPSED pickup).**
- ✅ **Seeding fixed the cold-start wall**: `nav/reach_cherrygrove` climbed 0→1.0 by ~8M (vs 066's stuck 0.02).
  The state diversity from 065's 400 seeded carry states bootstrapped cold navigation. `front` delivered 351×.
- ✅ **Pickup was DEVELOPING**: `nav/reach_route30_gate` reached **0.85** — start envs heading north past
  Cherrygrove toward Mr.Pokemon's (the pickup approach) — then ❌ **COLLAPSED to 0** (reach_cherrygrove 0.93→0),
  pickup never landed. The 6 carry-mode (red-marker) frontier envs' southward gradient overwhelmed the northward
  (black-marker) pickup behavior as it formed. **The marker did NOT separate the modes strongly enough at 6:6.**
- **Lesson**: seeding is the cold-nav-bootstrap key (keep it); 6 carry-frontier envs over-bias south. → 068
  rebalances to 3/9. **OPEN QUESTION the marker must answer: if 3/9 ALSO collapses pickup, the egg-marker is
  fundamentally insufficient to separate the modes → escalate (different representation or re-scope).**

## Agent 068 — SEEDED cold synthesis, REBALANCED 3/9 frontier, 2026-06-14

Single change vs 067: FRONTIER_N_ENVS 6→3 (the proven 062 ratio). Keep the SEEDING (the nav-bootstrap key) so
3 seeded frontier envs still supply carry diversity, while 9 start envs protect the pickup mode from
carry-dominance. Else = 067: COLD, EGG_MARKER, dense +0.05/carry-tile, seeded from 065's 400 cells, lr 1.5e-4.
- **Boot**: SEEDED 400 cells, frontier 3/12, cold, no errors.
- **THE test**: does the marker let pickup (black, 9 envs) and carry-nav (red, 3 seeded envs) COEXIST?
- **Gates**: ~10M `nav/reach_cherrygrove` → 1 (seeding bootstraps nav again); ~25M `nav/egg_received` rising
  AND HELD (pickup develops AND survives — the 067 collapse fixed); ~40-50M COEXISTENCE: pickup ≥0.5 AND
  carry-depth >1 → `nav/egg_delivered` off 0. ABORT ~25M if pickup develops then collapses AGAIN (marker
  insufficient even at 3/9 → escalate: the representational fix doesn't hold; re-scope or new approach).

**Run outcome (ABORTED ~16M — pickup developed to 0.58 then COLLAPSED to 0.03; marker insufficient even at 3/9).**
Same pattern as 067, only delayed: nav developed (reach_cherrygrove 0.93, egg_received 0.58) then BOTH collapsed
(reach_cherrygrove→0.13, egg_received→0.03). The 3/9 ratio delayed but did not prevent the carry-mode
overwhelming the pickup-mode. **The egg-marker is confirmed insufficient: a single shared CNN gets dominated by
the higher-gradient mode regardless of a visual mode-marker.** Then the firm pre-registration fired → tested the
hard-gated MIXTURE with the right teachers (the cheap alternative to re-scoping).

## SESSION CONCLUSION (2026-06-15): from-start egg delivery is a FUNDAMENTAL wall — comprehensive escalation

> Gated-mixture tests with the session's best teachers (053 pickup + 065 carry-nav):
> - 065@20M alone from mid_route30 (Cherrygrove carry): **delivered 0/12** (wanders NORTH, like 062).
> - GATED 053+065 from start.state: **delivered 0/20**.

**THE definitive finding — "carry-nav" was always a frontier-reset ARTIFACT.** Every run whose training metrics
suggested delivery progress (060-062 front/egg_delivered ~0.8; 065 carry-depth 4) was MEASURING frontier
episodes that RESET at/near the lab — never the policy NAVIGATING the Cherrygrove→Elm backtrack. In EVAL from a
standing carry state, NO policy across 059-068 delivers. The eval-time carry-navigation skill was never learned.

**What the session ruled out (the whole hypothesis space for from-start delivery, exhausted):**
1. Frontier reset / Go-Explore (059-063): discovers the near-lab A-press, never transfers carry-nav to start;
   erodes pickup. The "discovery" was an artifact.
2. Reward shaping — sparse (064): fails + destabilizes; dense (065): shifts the policy south but COLLAPSES pickup.
3. Representational egg-marker — warm (063) and cold (066-068): does NOT separate pickup/carry modes; the
   higher-gradient mode dominates the shared CNN and collapses the other (proven from both directions).
4. Hard-gated mixture of specialists (064 with 062, now with 065): fails because NO policy is an eval-robust
   carry-from-Cherrygrove specialist — that skill simply does not exist in any checkpoint.
ROOT (deepest): a strong NORTHWARD policy habit + a barren (zero-income, already-visited) southward corridor +
the pickup⟺carry mode tension in a shared CNN. Carrying the egg SOUTH from a standing start is the unsolved core.

**RECOMMENDED PATH (needs human decision — multi-run / new method):**
A. **Gated mixture of specialists, but BUILD the missing carry specialist first** — a dedicated RL run trained
   FROM carry states (mid_route30 / newbark_egg) with the delivery reward, FIXED-start so it must navigate the
   FULL Cherrygrove→Elm backtrack every episode (no frontier-reset-near-lab artifact) → a robust deliver-from-
   carry policy. Then gate 053 (pickup) + carry-specialist (egg flag) → deliver from start. Fixed-start training
   segregates, but for a GATED specialist that's FINE (no transfer needed). Most promising; ~1-2 runs.
B. **Goal-conditioned / hierarchical RL** — explicit sub-goal (the lab) with goal-conditioned value, or a
   manager that switches sub-policies. Addresses the root (the policy needs an explicit "go to the lab" goal)
   but a larger build.
C. **RE-SCOPE** — accept from-start delivery as the project's hard limit (now thoroughly characterized) and
   pursue PHASE 2 (Falkner) from delivered states, or redefine the success gate.
RECOMMENDATION: A (build a carry specialist, then gate) — directly fills the one missing piece the gated-mixture
needs, and it's the cheapest path to an actual from-start delivery. PAUSED for Fabio; GPU idle. The session's
tools (eval_gated_mixture.py, frontier archive + seeding, egg_marker gate, dense reward) are all reusable.

**HUMAN DECISION (2026-06-15): option A — build a carry specialist, then gate. → agent_069 LAUNCHED.**

## Agent 069 — CARRY SPECIALIST: deliver-from-carry, fixed-start (2026-06-15)

Build the one missing piece the gated mixture needs: a policy that ROBUSTLY navigates the egg from a standing
carry state to Elm (no frontier-reset-near-lab artifact). Train FIXED-start from carry states so it MUST learn
the full backtrack every episode.
- **Config**: curriculum 8×mid_route30 (Cherrygrove, egg in hand — the hard barren-gap backtrack) + 4×newbark_egg
  (New Bark, egg in hand — easier near-lab, helps the +30 reward propagate back). Warm 053 (map-nav base). NO
  frontier, NO marker (single carry mode — separation problem gone). Dense southward reward + breadcrumbs +
  delivery +30 teach south. ent 0.02 (explore past 053's N-habit), lr 1.5e-4. Boot OK (warm 053, no frontier).
- **Why this can succeed where everything else failed**: fixed-start = the agent practices the SAME
  Cherrygrove→Elm decision every episode with 12 envs → dense consistent gradient to overcome the northward
  habit (from-start never gave this — it rarely reached the carry state). Segregation is irrelevant (gated
  specialist, no transfer to start needed). Single mode (always carrying) → no pickup⟺carry collapse.
- **Metrics note**: all envs are carry states (is_start_env=False) → episodes log under `front/`. Watch
  `front/egg_delivered_rate` (the specialist's delivery rate).
- **Gates**: ~10M `front/egg_delivered_rate` rising (newbark_egg envs deliver first — near lab); ~25M ≥ 0.5
  (mid_route30 Cherrygrove backtrack learned); then EVAL from mid_route30 ≥ 0.5-0.8 (robust deliver-from-carry,
  the thing 062/065 got 0/12) → if good, the GATED 053+specialist test for from-start delivery. ABORT ~20M if
  `front/egg_delivered` still ~0 (even fixed-start can't learn the backtrack → the carry-nav is harder than a
  navigation problem; escalate to goal-conditioned RL or re-scope).

**Run outcome (ABORTED ~24M — CARRY SPECIALIST FAILED; even fixed-start can't beat 053's northward habit).**
`front/egg_delivered` dead-flat **~0.06** for 24M (6ths 0.056/0.078/0.044/0.047/0.076/0.066); `front/return_progress`
stuck at the start-average ~2.8 — the agents NEVER navigate south past their carry starts. The ~0.06 is just the
near-lab newbark_egg envs occasionally delivering; deliver-from-Cherrygrove (the actual backtrack) ≈ 0. The
bottleneck is NOT the A-press — the agents won't navigate south AT ALL. Warm-053's northward habit dominates even
a dedicated fixed-start specialist with delivery +30 + dense reward + ent 0.02.

## SESSION-2 CONCLUSION (2026-06-15): Cherrygrove→Elm carry-navigation has resisted EVERY method — re-scope?

> 11 runs this session (059-069). The egg's southward carry-navigation from a standing start is unsolved by:
> frontier reset (059-063) · reward shaping sparse+dense (064-065) · egg-marker warm+cold (063,066-068) ·
> gated mixture (064 w/062, w/065) · dedicated fixed-start carry SPECIALIST (069). The training "successes"
> were frontier-reset-near-lab ARTIFACTS; in EVAL, no policy delivers from a standing carry state.

**The wall, fully characterized**: a CNN policy warm or cold has a NORTHWARD navigation prior (all exploration
income was northward) and will not navigate the egg SOUTH across the barren (zero-income, already-visited)
Cherrygrove→Route29→Elm corridor — not with shaping, not with a mode-marker, not even when FORCED to from a
fixed carry start. The +30 delivery reward is too distal and the southward path too unrewarding to overcome the
prior within practical compute. This is the same wall ~13 agents have now hit; this session proved it is not a
reward-shaping or representation problem but an EXPLORATION/PRIOR problem.

**Remaining options (honest):**
- **RE-SCOPE (now strongly recommended)** — the from-start Zephyr badge has resisted 69 agents; the Phase-1
  delivery sub-wall is now exhaustively characterized. Pivot to PHASE 2 (Falkner) from delivered states (a
  fresh, possibly-more-tractable problem), OR redefine the success gate (e.g. deliver-from-mid_route30, which a
  specialist could plausibly reach with more work), OR declare the diagnostic the deliverable.
- **COLD carry specialist** (last technical lever for Phase 1) — train the specialist from carry states with NO
  warm-053 (so no northward habit) + a reverse curriculum (before_elm → newbark → mid_route30, A-press-first).
  Cold from a carry start is closer to the goal than cold-from-start, so the bootstrap risk is lower. Uncertain;
  another ~hours-long run; the prior may re-form cold too.
- **Goal-conditioned / hierarchical RL** — a fundamentally different method (explicit lab goal). Largest build.
RECOMMENDATION: RE-SCOPE to Phase 2 (or redefine the gate). We have characterized this wall as thoroughly as is
useful; further from-start-delivery attempts are diminishing returns. PAUSED for Fabio; GPU idle.

**HUMAN DECISION (2026-06-15): RE-SCOPE to PHASE 2 (Falkner). Phase-1 from-start delivery is the project's
characterized hard limit; pursue the gym fight from delivered/post-gate states.**

## PHASE 2 KICKOFF — beat Falkner from the gym (re-scoped success gate)

New goal: from `violet_city_gym.state`, beat Falkner (Zephyr Badge / EVENT_BEAT_FALKNER 0xD84E bit5). Segregation
no longer constrains us (we've dropped the from-start requirement), so a fixed gym state is the legitimate
training distribution. Diagnostic of the gym state: map (10,7), Totodile **lv15** (already strong enough vs
Falkner's Pidgey lv7 / Pidgeotto lv9), egg delivered, badge 0, gym_trainers 0. **So Phase 2 is a FOCUS problem,
not a strength problem**: per the agent_050 eval (W/F/L 288/38/1, 0/6 badge), the agent WANDERS OUT of the gym to
grind wild battles instead of completing the Falkner fight (the gym interior has no wild battles).

## Agent 070 — PHASE 2 Run 1: Falkner fight, focused gym training, warm 050 (2026-06-15)

12×violet_city_gym.state, warm agent_050@10M (the ONLY policy to ever beat Falkner — ~6× in training; its
fight competence is in the weights). Existing reward stack drives the fight: gym damage (+3×ΔenemyHP,
map-constrained to (10,7)), per-trainer-beaten (+5), gym-exit (+150), badge/zephyr (+30). NO frontier, NO
marker, ent 0.02 (explore the fight), lr 1.5e-4. Bet: focused 12×gym training CONSOLIDATES 050's rare Falkner
wins into a reliable badge_rate (all 12 envs in the gym → the fight is the dominant experience, vs 050's mixed
ladder where the gym was 1 of 6 rungs). Metrics log under `front/` (gym state ≠ start.state).
- **Gates**: ~10M `front/gym_trainers_mean` rising (beating the 2 gym trainers) + in-gym wins; ~25M
  `front/badge_rate` rising off 0 toward the milestone (≥ 0.5). ABORT/adjust ~15M if the agent WANDERS OUT
  (in_battle in wild, badge_rate flat) → run 2 cuts the exploration reward's wander-out pull / penalizes leaving
  the gym map, or starts from a state right in front of Falkner.

**Run outcome (ABORTED ~7.5M — naive training ERODES 050's Falkner capability; baseline established).** Warm-050
from the gym started at front/badge_rate 0.31 (training rollouts) but training ERODED it 0.31→0.03 as the policy
drifted back to wild-grinding (battles W/F ~39/54 per ep). **Clean eval of 050@10M from violet_city_gym.state
(stochastic, 20 eps, 25k cap): badge 2/20 (10%), battles W/F/L 417/31/1, reached Violet 20/20.** So 050's TRUE
Falkner capability from the gym is ~10%, and its dominant behavior is leaving the gym to grind ~21 wild wins/ep.
- **Phase-2 reframe**: this is a CONSOLIDATE-without-eroding problem (10% → ≥50% milestone), not build-from-zero.
  The blocker is the WANDER-OUT: exploration reward (+0.02/tile, +1.0/new map) + wild-battle income lure 050 out
  of the gym (no wild battles inside) → it grinds instead of fighting Falkner, and naive training amplifies this.
- **Run-2 plan (the clear next step, for human review)**: keep the agent IN the gym so the Falkner fight is the
  only income → it consolidates. Cleanest lever = CUT the exploration reward for this Phase-2 run (in the gym
  the agent needs to FIGHT, not explore for tiles); optionally a small off-(10,7) leave-gym penalty; warm 050,
  LOWER lr (~5e-5) + ent 0.01 to PRESERVE 050's 10% base and consolidate rather than erode. Gate: front/badge_rate
  rising past 0.10 toward 0.5, wild wins/ep dropping.

### Phase-2 status (2026-06-15)
Phase 1 = exhaustively characterized fundamental wall (re-scoped). Phase 2 = baselined: 050 is ~10% Falkner from
the gym; the path is to stop the wild-grind wander-out and consolidate.

## Agent 071 — PHASE 2 Run 2: consolidate Falkner, exploration CUT (2026-06-15, Fabio: "do what you recommend")
Single change vs 070: `EXPLORATION_SCALE=0` (new code: env→compute_reward param) zeros the new-tile/new-map
reward so the agent has NO incentive to leave the gym to grind — the Falkner FIGHT (gym damage +3×Δ, trainer +5,
badge +30) is the ONLY income. Warm 050@10M (its 10% Falkner capability), LOWER lr 5e-5 + ent 0.01 to PRESERVE
that base and consolidate (vs 070's lr 1.5e-4 which eroded it 0.31→0.03). 12×gym, no frontier/marker. Boot OK.
- Verified `exploration_scale=0` zeros reward_exploration (test). Phase-1 default stays 1.0 (untouched).
- **Gates**: ~8M wild wins/ep DROPPING (no longer leaving the gym to grind) + agent stays on (10,7); ~20M
  `front/badge_rate` rising past 0.10 toward the milestone (≥0.5). ABORT if badge_rate erodes below 0.05 again
  (cutting exploration insufficient → add a leave-gym penalty, or start in front of Falkner).

**Run outcome (ABORTED ~15M — exploration-cut PRESERVED 050's 10% but did NOT consolidate; wander persists).**
`front/badge_rate` oscillated ~0.04-0.13 (hovering at 050's 0.10 baseline, NOT rising toward 0.5 — low lr
preserved it, no 070-style erosion, but no improvement). `front/reach_violet` 1.0 + ~30 wild wins/ep every
episode — the agent STILL leaves the gym to grind despite zero exploration reward (the wild-WIN reward +2 +
050's wandering habit pull it out). **Lesson: cutting exploration is necessary but insufficient; the wander-out
needs a STRONGER lever.** Next (for human review): (a) leave-gym penalty (−/step off (10,7) — but penalties have
a stalling history here), (b) BUILD a start-in-Falkner-battle save-state (run 050 until enemy = Falkner's lv7/9
bird, save → train from inside the fight, forcing it, no wander option, no risky penalty) — RECOMMENDED.

### Phase-2 status @ 071 (paused, 2026-06-15)
050 is ~10% Falkner from the gym; cutting exploration preserves but doesn't consolidate it because the agent
keeps wandering out to wild-grind. The robust fix is to remove the wander option entirely (train from inside the
Falkner battle). PAUSED for Fabio to pick the next lever (forced-fight state vs leave-gym penalty vs wrap up).
GPU idle. EXPLORATION_SCALE knob (default 1.0) added and kept.

## Agent 072 — PHASE 2 Run 3: FORCE THE FIGHT (train from inside the Falkner battle) (2026-06-15, Fabio's pick)
Created `saves/falkner_battle.state` (ran 050 from the gym until its opponent was Falkner's Pidgeotto lv9; saved
at: Totodile lv15 @ FULL HP, in_battle vs Pidgeotto lv9). 12×this state, warm 050@10M, lr 1e-4, ent 0.02 (explore
the battle-menu move selection), exploration cut. The agent starts IN the fight → no wander option → it MUST
select attacks and win → badge (+30). lv15 vs lv9 @ full HP is heavily favored, so it should consolidate FAST if
it just learns to attack (vs flee / wrong menu). Boot OK.
- **Gates**: ~3-5M `front/badge_rate` rising fast toward ≥0.8 (the agent learns to KO Pidgeotto). If it sticks
  low, the issue is move-selection/menu navigation (the deep fight skill) → diagnose with --watch. If it goes
  high → SUCCESS on the isolated fight; extend backward (reverse curriculum: earlier in the Falkner fight, then
  the gym entrance) toward badge-from-the-gym-state.

**Run outcome (ABORTED ~9M — PROVED the fight is solvable (badge 0.49) then DESTABILIZED to 0).** `front/badge_rate`
0.10→0.49 (by ~5M) then collapsed 0.49→0 by 9M; `battles_lost` 0.07→0.66 (started DYING to Pidgeotto despite
lv15 vs lv9). The forced-fight state WORKS (isolated Falkner fight is winnable + learnable) but lr 1e-4/ent 0.02
over-trained past the winning move-policy into bad moves. → agent_073 stabilizes.

## Agent 073 — PHASE 2 Run 4: STABILIZE the Falkner fight (2026-06-15)
Warm from 072@5M (the ~0.49 badge peak) + lr 5e-5 + ent 0.01 (exploit the winning move, less drift). Same
12×falkner_battle.state, exploration cut. Gate: ~5M badge_rate HOLDS ≥0.4 and climbs toward ≥0.8 (stable fight);
if it collapses again → the move-policy is fundamentally unstable (diagnose moves via --watch). If it holds high →
isolated Falkner fight SOLVED → reverse curriculum backward toward badge-from-gym.

**073 outcome (ABORTED ~5M — badge 0):** warmed from 072@5M, but that checkpoint was PAST the 0.49 peak (already
degrading), so it started at ~0 and stayed. Lesson: the 0.49 was transient and no checkpoint sits exactly on it.
→ agent_074: warm 050 + lr 5e-5/ent 0.01 FROM STEP 1 (consolidate the winning move stably, never over-train).

## Agent 074 — PHASE 2 Run 5: stable Falkner fight from 050 (2026-06-15)
Warm 050@10M, lr 5e-5, ent 0.01, 12×falkner_battle, exploration cut. Low lr/ent from the start so it climbs to
the winning Pidgeotto move-policy WITHOUT 072's over-train collapse. Gate: ~6M badge_rate rising AND HOLDING ≥0.5
toward ≥0.8 (stable solve). If it can't hold (collapses like 072) → the battle move-selection is fundamentally
unstable under PPO → diagnose with --watch / consider a different fix. If stable → isolated Falkner fight SOLVED.

**074 outcome (ABORTED ~4.5M — low lr/ent too gentle to LEARN; badge stuck ~0.08).** The dilemma: 072's lr
1e-4/ent 0.02 LEARNED the fight (0.49) but destabilized; 074's lr 5e-5/ent 0.01 is stable but never climbs
(stays at 050's ~0.10 baseline). Plus a CONFUSING signal across 073/074: ~25 battles WON per episode with badge
~0.08, from a SINGLE-battle start state — shouldn't be possible if beating Pidgeotto ends the episode with the
badge. Needs --watch diagnosis (what does the agent DO from falkner_battle.state?), not more blind lr tuning.

## PHASE 2 STATUS — paused after a long session (2026-06-15)
- Phase 1 (from-start egg delivery): exhaustively-characterized FUNDAMENTAL WALL (re-scoped). 11 runs ruled out
  frontier reset, reward shaping, egg-marker, gated mixture, dedicated carry specialist.
- Phase 2 (Falkner from gym): baselined (050 = 10% badge, wild-grinds). The forced-fight state (072) PROVED the
  isolated Falkner fight is winnable+learnable (badge → 0.49) — a real positive — but the training is UNSTABLE
  (over-trains into bad moves) and there's an unexplained 25-wins-badge-0.08 dynamic.
- **NEXT (needs --watch / human): (1) diagnose what the agent does from falkner_battle.state (why 25 wins, why
  badge doesn't register/hold); (2) find a stable fight recipe (e.g. lr ~7e-5 + CHECKPOINT_FREQ lowered to catch
  the peak + early-stop, or reward the KO of Falkner's birds specifically, or a curriculum from the battle's
  first turn); (3) then reverse curriculum backward toward badge-from-gym.**
PAUSED; GPU idle. This was a very long session — handing back a clean status; Phase 2 is close (fight winnable)
but needs visual diagnosis of the battle dynamics before the next training recipe.

## Agent 075 — PHASE 2 Run 6: STABILIZE the Falkner fight (learn THEN commit) (2026-06-15)
Targeted fix for the 072/074 dilemma (072 lr 1e-4 learned 0.49 then drifted into bad moves; 074 lr 5e-5 too
gentle, stuck ~0.08). Recipe: warm 050@10M, lr **7e-5** (intermediate) + **entropy SCHEDULE 0.02→0.005 over 8M**
(EntCoefScheduleCallback) so it explores the battle-menu move EARLY then COMMITS to the winning one (stops the
drift), + **CHECKPOINT_FREQ 1M** to capture the badge peak (072's 0.49 fell between 5M-spaced saves). Same
12×falkner_battle, exploration cut, no frontier/marker. Boot OK (ent_coef_sched 0.02).
- **Gates**: ~6M `front/badge_rate` rising past 0.4 AND, as ent anneals (~8M), HOLDING/climbing toward ≥0.8
  (stable solve — no 072-style collapse); keep the best 1M-checkpoint regardless. If it still collapses as ent
  drops → the move-policy is intrinsically unstable under PPO (→ would need a denser reward, e.g. reward the KO
  of Falkner's birds, or accept and move to the LLM agent). If stable → isolated Falkner fight SOLVED; pick the
  best checkpoint and (optionally) reverse-curriculum backward toward badge-from-the-gym.

**Run outcome (STOPPED ~24.7M — NOT stabilizable, but the 1M checkpoints captured a usable peak).** Training
`badge_rate` peaked ~0.49 (~3-4M) then COLLAPSED to 0 and stayed there to 24.7M, even at ent 0.005 — confirms
the Falkner move-policy is INTRINSICALLY UNSTABLE under continued PPO (the entropy schedule did not prevent it).
**But the 1M checkpoints worked: stochastic eval (20 eps from falkner_battle.state) — @1M 10%, @2M 25%, @3M 0%,
@4M 0%.** → **Phase-2 deliverable = `agent_075_2999988` (2M), ~25% badge from the in-battle state** (W/F/L 34/0/0:
wins Pidgeotto 1-in-4, never flees/dies). Training badge_rate (0.49) > eval (25%) — the train rollouts were
optimistic; 25% is the honest figure.

### PHASE 2 — final status (2026-06-15)
The isolated Falkner fight is WINNABLE (~25% eval, best checkpoint agent_075@2M) but NOT reliably solvable with
PPO from pixels — the move-policy won't hold. Pushing further on the fight needs either a DENSER fight reward
(reward KO of Falkner's Pidgey/Pidgeotto specifically, so the signal isn't the sparse end-of-fight badge) or a
fundamentally different method. **Recommendation: this is the practical ceiling of RL here; bank agent_075@2M as
the Phase-2 result and pivot to the LLM agent (Phase 5), OR do one fight-reward-redesign run if a reliable
"Falkner beaten" is required for the video.** GPU idle.

## RL PHASE CLOSED (2026-06-15, Fabio's decision)
The RL line of the project is concluded. Final results, banked:
- **Phase 1** (egg delivery from start.state): an exhaustively-characterized FUNDAMENTAL WALL — an exploration/
  prior problem, not reward/representation. Best policy: `agent_053` (start → Cherrygrove → Mr.Pokemon's →
  pickup at 100%; never delivers). Whole hypothesis space ruled out (frontier reset, shaping, egg-marker, gated
  mixture, carry specialist).
- **Phase 2** (beat Falkner): the fight is WINNABLE but not reliably solvable with PPO-from-pixels. Deliverable:
  **`agent_075_2999988` (2M) — 25% badge** from `saves/falkner_battle.state` (stochastic, 20 eps).
- Badge from a true start.state: NEVER achieved across 70+ agents — now thoroughly understood.
- **NEXT (new chapter, Fabio will start it): the LLM agent (planned Phase 5)** — reason about the quest with
  prior knowledge, to contrast "learn from scratch (RL)" vs "reason (LLM)". Reusable assets: `make_gif.py`,
  `eval_gated_mixture.py`, `saves/falkner_battle.state`, `docs/PROJECT_NARRATIVE.md` (video/blog script),
  and all checkpoints. No further RL runs planned. GPU idle.

---

## RL v2 — RE-OPENED (2026-06-23, Whidden-inspired re-baseline, Fabio's decision)

After re-watching PWhiddy's video, RL was re-opened with a SIMPLIFIED objective (he skipped the early
backtracking; we skip it too). New baseline: ONE generalist trained from **`saves/egg_delivered_clean.state`**
(New Bark, egg already delivered → Route-30 gate already open), goal = navigate New Bark → Violet City →
**beat Falkner**, no backtracking. New tooling: offline map-visualization overlay (`agents/rl/visualize_map.py`),
Dockerized playback (`agents/rl/play.py`). The carry/return reward logic is **self-inert** from this state
(egg pre-delivered), so no reward surgery — just re-enable forward exploration.

## Agent 076 — v2 cold generalist from egg-delivered (2026-06-23)
- Cold (`INIT=None`), `EXPLORATION_SCALE=1.0` re-enabled, `12×egg_delivered_clean`, no frontier, 15M validation.
- **Outcome (stopped ~9M): the Route-29 wall, confirmed.** `nav/reach_cherrygrove` flat **0.0** through 8.7M;
  rollout = 89% on Route 29 / 11% New Bark, never reaches Cherrygrove — the agent grinds Route-29 wild battles.
- **Lesson**: a cold pure-start (even from egg-delivered) does NOT learn directed forward navigation — same
  family as Agent 019/066. More steps alone reinforce the grind; needs STATE DIVERSITY.

## Agent 077 — v2.1 + frontier (Go-Explore), no seed (2026-06-23)
- Frontier ON (4/12 envs), `FRONTIER_SEED_FROM=None`, else = 076 (explore 1.0).
- **Outcome (stopped ~15M): FAILED the same way.** Archive stuck at ~72 cells (only New Bark + EAST Route 29);
  `nav/`+`front/reach_cherrygrove` both 0. The frontier resets only into cells it has already discovered, so
  it **amplifies the reachable region but cannot CROSS the Route-29→Cherrygrove bottleneck unaided**.
- **Lesson**: frontier without seeding ≠ a fix for a bottleneck. It needs diversity placed PAST the wall
  (seeding, or curriculum anchors) — consistent with Agent 067's seeded result.

## Agent 078 — v2.2 forward curriculum (Violet anchors) + bidirectional frontier (2026-06-23)
- `CURRICULUM_STATES_CNN = 3×violet_city + 2×violet_city_gym + 7×egg_delivered_clean` (last 4 = frontier envs),
  explore 1.0. Idea: Violet anchors seed the FORWARD end so the shared frontier has cells past the bottleneck.
- **Outcome (stopped ~15M): partial.** Archive bridged from BOTH ends (Route 29 start-side + Violet/Route-31
  anchor-side, 634 cells) but the MIDDLE stayed empty (Cherrygrove `26_3` = 0); `nav/reach_cherrygrove` still 0.
  The two fronts approached but didn't meet. `front/badge_rate` ~0.7 (gym anchors trivially beat Falkner — an
  ARTIFACT of starting at the gym, not navigation).
- **Lesson**: anchors + frontier populate the archive but don't, by themselves, drive the START policy across
  the gap; needed a stronger forward pull.

## Agent 079 — v2.3 = v2.2 + EXPLORATION_SCALE 4.0 (2026-06-23) — BREAKTHROUGH then PLATEAU
- Cold, explore **4.0** (a new-MAP crossing now out-earns the Route-29 grind), else = 078. Budget 480M (~2 days).
- **BREAKTHROUGH ~13-15M**: the corridor archive connected **end-to-end** (New Bark→Route29→Cherrygrove→Route30→
  Route31→Violet Gatehouse→Violet→Gym), and — for the FIRST TIME ever — the START policy advanced past the
  Phase-1 wall, **segment by segment**: `reach_cherrygrove` consolidated to 1.0 ~28M, `reach_route30_gate` to
  1.0 ~37M (pace accelerating: ~14M then ~9M/segment).
- **PLATEAU at Route 30**: `nav/reach_route31` stayed **0.0 for ~83M** (44M→127M, with a noisy dip+recovery on
  route30_gate). Stopped ~130M.
- **Diagnosis (Fabio's insight)**: `EXPLORATION_SCALE=4.0` rewards entering ANY new map — including **Dark Cave
  & other dead-end maps** (235 cave/bank-3 cells harvested! 2nd-most after the corridor) and Violet's Sprout
  Tower / school. Each cave floor / tower floor is a fresh "+0.4 new map", LURING the agent off the path; it
  burns the episode in caves instead of crossing into Route 31.
- **Lesson (big)**: bidirectional Go-Explore + a strong exploration scale CONNECTS the corridor and is the first
  method to push the start policy past Cherrygrove/Route 30 (vs the Phase-1 wall) — BUT a high exploration scale
  creates a NEW failure mode: dead-end-map farming. The exploration reward must be **corridor-restricted**.

## Agent 080 — v2.4 off-path exploration WHITELIST (2026-06-24) — RUNNING
- `env/rewards.py`: new `CORRIDOR_WHITELIST` — the exploration reward (new-tile + new-map) pays **only** on the
  ~8 corridor maps; caves, Sprout Tower, school, houses, side routes pay **0** (verified: corridor +0.408,
  off-path 0.0). Combat/level/heal/badge UNCHANGED (the gym fight already works). Warm from `079@130M`, 20M
  validation, frontier+curriculum kept.
- **Watching `nav/reach_route31`**: if it cracks > 0, the off-path lure WAS the Route-30 plateau cause → relaunch
  the long 480M run; if still 0 at 20M, the warm policy's habit is sticky / fix insufficient → cold rerun or a
  Route-31 curriculum anchor.

Dopo 18 run di MlpPolicy con badge=0/10

Dopo 18 run di MlpPolicy con badge=0/10 da start.state in ogni eval, il problema è **rappresentazionale**, non di reward shaping. Riferimento: PokemonRedExperiments (Peter Whidden) che ha risolto Pokemon Red con CnnPolicy + frame stack.

### Cambiamenti strutturali rispetto al filone MLP
- **Env nuovo**: `env/pokemon_env_cnn.py` (obs = screen ndarray invece di state vector)
- **Trainer nuovo**: `agents/rl/train_cnn.py`
- **Policy**: `"CnnPolicy"` (NatureCNN: 3 conv layers + 2 fc layers)
- **Observation space**: `Box(0, 255, shape=(72, 80, 3), uint8)` — screen downsampled 50%
- **Frame stacking**: `VecFrameStack(n_stack=4)` per dare informazione di movimento
- **Compute reward**: identico al filone MLP (riutilizzo da `env/pokemon_env.py`)
- **Ram reader**: invariato — la verità ground truth per i reward viene sempre dalla RAM
- **n_envs**: ridotto da 8 a 4 (CNN forward è più costoso)
- **n_steps**: ridotto da 8192 a 2048
- **learning_rate**: 2.5e-4 (vs 3e-4) — leggermente più conservativo
- **batch_size**: 256 (vs 64) — sfrutta la GPU
- **device**: `"cuda"` (per la prima volta — MLP era CPU-bound)
- **TOTAL_TIMESTEPS**: 50M per prima validazione (vs 200M MLP — un episodio CNN costa ~10×)

---

## Strategies To Try (After Agent 012)

### If Agent 012 succeeds (badge obtained from start.state):
- Valutazione quantitativa: badge rate su 100 episodi da start.state
- Passare all'agente LLM (Phase 5 del plan) per il confronto

### If Agent 012 fails (route_31 envs raggiungono gym ma start.state no):
- Aggiungere un secondo bridge (es. violet_city_west.state o un punto su Route 29)
- Oppure aumentare ulteriormente i per-episode waypoints per i primi segmenti del percorso

### Ruled out / do not retry
- Revisited tile penalty (Agent 002 lesson)
- visited_maps reset per episode (Agent 003 lesson)
- Map transition bonus > +30 per-lifetime (causes reward hacking — Agent 003 lesson; per-episode è ok se piccolo)
- Unverified Sprout Tower flags 0xD85C/0xD85D (wrong addresses)
- ent_coef < 0.05 with this reward scale (premature convergence)
- Map Card event reward (not in standard flag range, Cherrygrove +20 suffices)
- sprout_tower_2f in curriculum (isola narrativa, non contribuisce al path verso Violet City)
- 3×before_elm_delivery senza VecNormalize (troppa varianza di return, causa regressione — Agent 006 lesson)
- violet_city_gym.state nel curriculum SOLO per il milestone one-shot +400 (mappa già in visited_maps all'init — Agent 008 lesson). Riadottato in Agent 015 con peso 1 per battle training diretto — il one-shot non spara, ma damage reward e badge reward sì.
- Battle reward proporzionale a HP (catena causale troppo lunga, doppia lettura RAM nello stesso step sempre 0)
- Battle win reward flat +15 (crea local optima: grinding near start.state > navigare verso waypoint distanti — Agent 010 lesson)
- **Stuck penalty -0.02** (Agent 015 lesson) — penalty troppo aggressivo: 28,740 step × 0.02 = −575/ep vs +260 new tile reward. Ratio 2.2:1 contro l'esplorazione. Il training si blocca a −180 di ep_rew_mean da 130M step in poi. Usare −0.003 (breakeven < 0.009).
- **Damage reward in wild battles** (Agent 016 lesson) — `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0` indistinto tra wild/trainer/gym crea local optima: l'agente combatte wild Pokémon (reward immediato +5×delta per turno) invece di navigare verso il gym (reward distante). Risultato: `in_battle` smoothed = 0.124, badge=0/10 in eval. Rimosso in Agent 017. Se mai re-introdotto, deve essere map-constrained al gym (10,7).
- **Wild battle penalty -3.0/step** (Agent 017 lesson, fermato a 22%) — overshoot opposto del damage_reward: penalty troppo aggressivo crea stallo oscillante. 1 wild battle media (80 step) = −240; per attraversare Route 30+31 con 5-10 wild battle inevitabili: −1,200/−2,400 vs +455 waypoint reward → netto negativo per muoversi → agente preferisce stare fermo. Calibrato a -1.0 in Agent 018.
- **MlpPolicy + state vector** (filone Agent 001→Agent 018 chiuso 2026-06-03) — 18 run con tutte le combinazioni di reward shaping, curriculum, observation features. Mai badge da start.state in eval. Convergenza a "stable suboptimal" in Agent 018 con `explained_variance` 0.99+ e `policy_gradient_loss` ~0. Lo state vector non porta informazione spaziale sufficiente per piani di navigazione lunghi. Filone abbandonato a favore di CnnPolicy.
- **CnnPolicy senza curriculum diversity** (Agent 019 lesson, 2026-06-03) — 4×start.state convergeva a stable suboptimal a -500 in 4M step. Il problema non era solo rappresentazionale ma anche di propagazione del reward: il policy network ha bisogno di esempi "facili" dai curriculum envs per imparare associazioni stato→azione che generalizzano. Confermato che 8 envs misti + clip stretto + batch grande sblocca il break-even (Agent 020).
- **Damage reward map-constrained al gym da solo** (Agent 021 lesson) — inerte da start.state: l'agente non raggiunge mai il gym, quindi il segnale non propaga. In più non risolve l'attrattore "resta a New Bark" → in_battle=0% in eval. Va abbinato a un incentivo forte all'uscita.
- **Mega-bonus waypoint per-episodio (80×)** (Agent 022 lesson) — aumentare i waypoint a +300/+500 fa salire ep_rew_mean in training (i curriculum env li raccolgono) ma NON fa generalizzare la policy a start.state, e destabilizza il value (explained_variance 0.25). Eval peggiore di Agent 020/Agent 021. Sintomo di **policy segregation visiva**: la CnnPolicy impara policy distinte per aspetto visivo che non transferiscono. Limite strutturale del reward shaping con curriculum eterogeneo.
- **Reward shaping per risolvere policy segregation** (filone Agent 020→4 chiuso 2026-06-05) — tre run consecutivi 0/10 badge da start.state. Calibrare i reward (wild penalty, damage, waypoint) cambia il comportamento dei curriculum env ma non fa generalizzare a start.state. Serve un segnale di **novelty intrinseco visivo** (KNN frame embedding) invece di waypoint hardcoded. → filone KNN (Agent 023).
- **Fixed-anchor / curriculum injection of the egg delivery** (Agents 047–051, 055–057, closed 2026-06-13) — every mix of `before_elm_delivery`/route/gym save-states with `start.state`, at every base (053 lineage, 050), entropy (0.01–0.03), anchor count (1–6), and win cap (2–10), SEGREGATED the start policy: the anchors deliver/grind at ~1.0 while `nav/egg_received` stays pinned (057: 0.0 for 5M). A fixed foreign save-state is a visual island the shared CNN keys a separate sub-policy to; it does not transfer to `start.state`. Do not add ANY fixed reset state to "inject" a skill.
- **Pure-start reward/entropy/reset tweaks for the egg delivery** (Agents 053–058, closed 2026-06-13) — `RETURN_BREADCRUMBS` (positive latched southward shaping), the directional pickup tile-reset, and the `EntCoefScheduleCallback` (0.03→0.01) are all deployed together; `nav/egg_delivered_rate` left 0 exactly once (054 spike, unconsolidated) and was flat 0 in the cleanest run (058), which also de-locked pickup. The delivery is a never-practiced ACTION (the Elm A-press) that no reward gradient or entropy setting can synthesize from a policy that reaches the frontier too rarely. Next attempt must be the START-CONTINUOUS frontier-reset architecture (design proposal above), not another reward/entropy run.

---

## Future Upgrades — Ispirate a PokemonRedExperiments

Backlog ordinato per impatto stimato. Da provare in questo ordine se Agent 013+ non raggiunge il badge da start.state in modo consistente.

### Già implementate
- [x] Lead Pokemon level in obs (Agent 013) — 0xDA49
- [x] Heal reward (Agent 013) — +30 quando hp_ratio aumenta >0.4 fuori battaglia
- [x] MAX_STEPS 4× (Agent 013) — 2^16 = 65,536
- [x] Step penalty ridotta (Agent 014) — -0.01 → -0.001
- [x] Opponent level in obs (Agent 014) — 0xD0FC (DataCrystal verified, Falkner Pidgey=7 / Pidgeot=9 ✓)
- [x] Enemy HP ratio in obs (Agent 014) — 0xD0FF/D100 current, 0xD101/D102 max
- [x] N_STEPS 8192 (Agent 015) — 4096 → 8192. Un episodio da 65k step copre ~8 rollout invece di ~16 → meno errori di bootstrap accumulati per episodio.
- [x] Damage reward (Agent 015) — `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0`. ✓ formula corretta.
- [x] violet_city_gym.state nel curriculum (Agent 015, peso 0→1) — per training diretto su Falkner. One-shot +400 non spara (mappa già in visited_maps), per-episode +200 spara su re-entry.
- [x] Stuck penalty (Agent 015 → Agent 016) — introdotto a -0.02 in Agent 015, calibrato a **-0.003** in Agent 016. Con -0.02 il penalty (575/ep) dominava l'exploration reward (260/ep) → training stagnante a −180. Breakeven: penalty < 0.009/step. -0.003 porta il baseline a +315/ep (positivo). Diverso da Agent 002 (quello era -0.01/step flat sempre, senza distinzione new/old tile).
- [x] Event flags binari in obs (Agent 015) — 3 feature binarie agli indici 8-10: rival beaten, egg received, egg delivered. Obs dim 13→15 (sostituisce i 2 byte normalizzati).
- [x] Party levels tutti e 6 (Agent 015, dim 15→20) — indici 15-19: slot 2-6. Struct size=0x30 calcolato, addresses: 0xDA79/0xDAA9/0xDAD9/0xDB09/0xDB39. Verificare con test_enemy_level.py.
- [x] Gym battle exit reward +150 (Agent 015) — map-constrained (10,7). Fires su battle falling edge dentro gym. Max 3×/ep. Safe vs grinding (lesson Agent 010).
- [x] Wild battle penalty (Agent 017 → Agent 018) — introdotto a `−3.0/step` in Agent 017, troppo aggressivo (stallo oscillante a 22%). Calibrato a `−1.0/step` in Agent 018. Map-constrained: NOT (10,7). Math: 1 wild battle ≈ 80 step → costo −80 (era −240) → gradiente verso il gym diventa marginalmente positivo (+455 waypoint − 600 wild penalty ≈ −145 netto, vs −1,345 di Agent 017).
- [x] Damage reward rimosso (Agent 017) — eliminato `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0`. Era responsabile dell'attrattore wild battle.

### Livello successivo (se Agent 015 fallisce)
| Upgrade | Stato | Descrizione | Impatto stimato | Costo |
|---------|-------|-------------|-----------------|-------|
| **DataCrystal event tracking completo** | 🔲 da fare | Reward per ogni event flag da New Bark Town a Falkner. Gym trainer 1/2 flags da scoprire con `test_enemy_level.py start`. Nota: running shoes non esiste in Gen 2, Pokédex/Pokéball già coperti da ELM_BIT. | Alto | Medio (ricerca + verifica empirica) |
| **Image-based observation** | 🔲 ultimo resort | Switch da MlpPolicy + vettore a CnnPolicy + screen (72×80 RGB) come PokemonRed. L'agente vede il gioco come un umano. | Molto alto | Molto alto — refactoring completo, 10× più compute |

---

## Agent 083 — GYM TASK (vertical slice for the RL-vs-LLM comparison)

**Started 2026-06-29.** Scope pivot: instead of the full corridor (the characterized exploration
wall), both RL and the LLM agent now target the **gym vertical slice** — start inside the Violet
City Gym, navigate to Falkner, win the Zephyr Badge. This makes the comparison achievable for both.

**Config (vs 082):**
- `CURRICULUM_STATES_CNN = [("saves/violet_city_gym.state", 12)]` — all 12 envs start inside the gym.
- `EXPLORATION_SCALE = 0.0` (anti-wander, per 071), `FRONTIER_ENABLED = False`.
- `INIT_FROM_CHECKPOINT = agent_050@10M` (gym-capable base; 072 hit badge_rate 0.49 warming from it).
- Hyperparams = 075's gym-stabilize settings (lr 7e-5, ent schedule 0.02→0.005 over 8M).
- TensorBoard: `runs/agent_083_2`. Checkpoints: `runs/checkpoints/agent_083/` (every 1M steps).

**Early observation (~0.5M steps):** `battles_won_mean ≈ 50`, `gym_trainers_mean` 0→1, but
`badge_rate = 0`. The agent is wandering OUT of the gym to wild-grind battle wins instead of
completing the Falkner fight — the documented 070-071 failure mode (win reward pays even with
EXPLORATION_SCALE=0). Watching `front/badge_rate`: if it stays 0 past a few M steps, switch the
start state to `saves/falkner_battle.state` (the 072 fix: forces the fight, no wander option), or
mix gym + falkner_battle envs.

**Result (20M, completed):** the warm-050 policy **transiently won** (peak ~5M steps), but the
train-time `front/badge_rate` is MISLEADING — with 131k-step episodes few terminate per rollout, so
a single badge-ending episode spikes it to 1.0. The honest **eval of the best checkpoint**
(`agent_083_4999980_steps.zip`, the 5M peak) over 10 independent episodes from `violet_city_gym.state`
gives **badge_rate = 10% (1/10)**: the agent wanders OUT of the gym (148-208 tiles, `end=step_cap` 9/10)
and farms wild battles (51 W / 0 L) instead of reliably fighting Falkner. Pure-gym start lets it
wander; documented fix (072) is to force the fight. **Lesson: never trust train-time badge_rate with
long episodes — always eval.** Pivoting to agent_084 (mixed curriculum) per the user's pre-approval.

**Best checkpoint:** `runs/checkpoints/agent_083/agent_083_4999980_steps.zip` (eval badge_rate 10%).

---

## Agent 084 — GYM TASK v2 (MIXED curriculum) — FAILED the gym-start task

**2026-06-29.** Pivot from 083: `CURRICULUM_STATES_CNN = 6×violet_city_gym + 6×falkner_battle`
(force the fight while keeping navigation), warm from `agent_083_4999980` (5M), 10M, EXPLORATION_SCALE=0,
frontier off. Logs `runs/agent_084_2`, checkpoints `runs/checkpoints/agent_084/`.

**Result — eval sweep from `violet_city_gym.state` (10 eps each):**
| Checkpoint | Badge | Battles W/F/L |
|---|---|---|
| 4M  | 0% | 17/1/0 |
| 7M  | 0% | 72/0/0 |
| 10M | 0% | 121/0/0 |
| final | 0% | 96/2/0 |

**WORSE than 083 (which hit 10% @5M).** The mixed curriculum reinforced battle-winning competence,
but from the gym START the agent wanders OUT and wild-grinds (121 wins/10 eps) instead of climbing to
Falkner. **Root cause (diagnosed):** with `EXPLORATION_SCALE=0` there is NO reward gradient pulling the
agent toward Falkner — tile income is off, and the gym-damage reward only fires INSIDE the Falkner
battle (never reached). The win reward is already capped at 2/episode (rewards.py:285) so grinding
doesn't *pay*, but with nothing pulling it up the gym, the agent just grabs its 2 capped wins nearby
and wanders. Adding falkner_battle envs can't fix a missing NAVIGATION signal.

**Recommended next step (reward engineering — user's call):** add a one-shot reward for *starting*
the Falkner fight (enter `battle_type>0` on `GYM_MAP`) to bridge the navigation gap, OR restore a
small gym-map-constrained exploration reward, OR accept the 072 pure-`falkner_battle.state` approach
(badge ~0.49) as the RL "battle-competence" baseline and let the LLM own navigation+battle.

**Best gym-start checkpoint to date:** `runs/checkpoints/agent_083/agent_083_4999980_steps.zip` (10%).

---

## Agent 085 — GYM TASK v3 (NEW "gym engage" reward) — BEST SO FAR (40% badge)

**2026-06-30.** Added a one-shot "gym engage" reward (`rewards.py`): +2.0 (capped 3) on the RISING
edge of a battle on `GYM_MAP`, supplying the navigation signal toward Falkner that 083/084 lacked.
PURE 12×violet_city_gym (one-variable test vs 083), warm from `agent_083_4999980` (5M), 10M,
EXPLORATION_SCALE=0, frontier off. Logs `runs/agent_085_2`.

**Result — eval sweep from `violet_city_gym.state` (10 eps each):**
| Checkpoint | Badge | Battles W/F/L |
|---|---|---|
| 2M | 10% | 14/1/0 |
| 3M | 10% | 11/1/0 |
| 4M | 0%  | 25/0/0 |
| **5M (`agent_085_4999980`)** | **40% (4/10)** | 31/0/0 |
| 7M | 0% | 57/0/0 |
| 10M | 0% | 54/0/0 |
| final | 0% | 74/0/0 |

**The engage reward WORKED** — 40% badge at 5M, **4× better than 083 (10%)**, and the right early
signature (battles_won≈3, gym_trainers=2 at ~3M = climbing the gym, not wandering). But the policy
**drifts back to wandering/grinding after 5M** (battles_won 54-74, badge 0) — the same learn-then-drift
instability as 083. Best checkpoint capped at 40%.

**Best gym agent to date:** `runs/checkpoints/agent_085/agent_085_4999980_steps.zip` (**40% badge**).

**Recommended next (to push toward reliable ≥0.7):** a 075-style STABILIZATION run — warm from
085@5M, lower lr (7e-5→5e-5) + lower/flat entropy (stop the 0.02→0.005 anneal that keeps exploring
off the winning policy), short (~5M). This should consolidate the 5M peak instead of drifting off it.

---

## Agent 086 — GYM TASK v4 (STABILIZATION) — FAILED to beat 085's 40%

**2026-06-30.** Stabilization run to consolidate 085's 40% peak: warm from `agent_085_4999980`,
lr 7e-5→5e-5, entropy LOW+FLAT (0.005, no anneal), pure 12×gym, engage reward kept, 5M.

**Result — eval sweep from `violet_city_gym.state` (10 eps each):** 1M=0%, 2M=0%, **3M=20%, 4M=30%**,
5M=0%, final=0%. **Best = 30% — WORSE than 085's 40%.** Even warm-starting from the 40% checkpoint,
the policy drifted back to wander/grind (battles_won ~48-62 late). Lower lr/entropy slowed but did
not prevent the drift.

**Diagnosis (3 runs converge on it): the wander-grind basin is the STABLE ATTRACTOR of this reward
landscape.** From `violet_city_gym.state` the agent CAN leave the gym; "exit + grab 2 capped wild
wins" is an easy, reliable basin, while "climb the gym + beat 2 trainers + beat Falkner" is a long
fragile sequence. Reward tweaks (engage reward: 10%→40%) and optimizer tweaks (stabilization: 30%)
move the needle but can't break ~40% — it's STRUCTURAL.

**Current RL baseline (best to date): `runs/checkpoints/agent_085/agent_085_4999980_steps.zip` = 40%.**

**Recommended STRUCTURAL fix (next, needs user decision):** add a penalty for LEAVING `GYM_MAP`
(or terminate the episode on leaving) so wandering out to grind is no longer an option — the agent
is then forced to solve the gym. Alternatively accept 40% as the RL gym baseline for the RL-vs-LLM
comparison and move to the LLM side.

---

## Agent 087 — GYM TASK v5 (STRUCTURAL FIX: confine-to-gym) — ✅ SOLVED (100% badge)

**2026-06-30.** Structural fix: new env flag `CONFINE_TO_GYM` (gated, off for the corridor) ends the
episode the moment the agent leaves `GYM_MAP` — removing the wander-grind OPTION that capped 083-086
at ~40%. WARM from `agent_085_4999980` (40%), pure 12×gym, engage reward kept, lr 5e-5, flat entropy
0.005, 5M. Logs `runs/agent_087_2`.

**Result — eval sweep from `violet_city_gym.state` (10 eps each, eval env does NOT confine — honest test):**
| Checkpoint | Badge | Avg steps | Battles W/F/L |
|---|---|---|---|
| 1M | 80% | 2468 | 27/0/1 |
| 2M | 100% | 1167 | 30/0/0 |
| 3M | 100% | 1052 | 30/0/0 |
| 4M | 100% | 840 ± 29 | 30/0/0 |
| 5M | 100% | 836 ± 23 | 30/0/0 |
| **final** | **100% (10/10)** | **839 ± 24** | **30/0/0** |

**SOLVED.** 40% → **100% badge**, reliable (10/10), efficient (~840 steps), zero losses, and STABLE
(no drift — the confinement holds the policy). The eval env does NOT confine, so the agent genuinely
learned to STAY in the gym and beat Falkner without the training crutch. `battles W/F/L = 30/0/0` =
exactly 3 per episode (2 bird-keepers + Falkner) — it does the clean gym sequence, no grinding.

**RL GYM AGENT (deliverable for the RL-vs-LLM comparison): `runs/checkpoints/agent_087/agent_087_final.zip`
— badge_rate 100%, ~840 steps/episode.**

**The arc (the lesson):** 083 pure-gym 10% → 084 mixed 0% → 085 +engage-reward 40% → 086 stabilization
30% → 087 +confinement **100%**. Reward/optimizer tweaks couldn't break the wander-grind basin; the fix
was STRUCTURAL — remove the bad option from the state space. Also reaffirmed: train-time badge_rate is
noisy with long episodes; always validate by eval.

---

## LLM agent — gym slice (2026-07-01)

**Setup:** local `qwen3-vl:8b` via Ollama, ReAct loop (RAM text + screenshot each turn), tools
`move`/`press`/`get_state`/`wait_frames`, from `saves/violet_city_gym.state` (map `(10,7)`).

**Result: fights but does not navigate. Badge rate 0% (RL is 100%).** The agent reliably starts and
wins the first bird-keeper battle (mash `a`), but never climbs to Falkner — it fixates at junctions and
funnels toward the gym exit.

Debug arc (systematic-debugging; each bug masked the next):
- **`settle` frames** (`pyboy_wrapper.step(settle=)`, LLM passes 24; RL default 0): consecutive presses
  had no released-button gap → no distinct button-down *edges* → dialogue/trainer scripts never
  advanced. Empirically ≥16 frames needed; without it, NO battle ever started.
- **Empty-response fallback** (`agent.py`): qwen3-vl returns an empty completion ~30–40% of steps;
  doing nothing self-perpetuates (same screen → same empty reply). Fallback: press `a`.
- **Walkability probe + anti-fixation guardrail** (`agent.py`): save/restore 4-dir probe reports which
  moves actually change position; a direction that failed to move is blacklisted for that tile. Broke
  the "push into a wall / A-spam" loops.
- **Gym confinement** (mirrors RL `CONFINE_TO_GYM`): undo any action that leaves map `(10,7)`. Needed a
  ROBUST restore — the naive `load_state(pre_snap)+tick(1)` intermittently *re-fired* a pending door
  warp (found via `[CONFINE] restored=False` logging). Fix: snapshot BEFORE the probe (clean resting
  state) + a verified in-gym anchor fallback + retry. Final run: 0 restore failures, map stayed
  `(10,7)`, 73 exits blocked — yet the agent still bounced at the door, `tiles` frozen, `won=1`.

**Lesson:** same shape as the RL fix (remove the bad option structurally), but structure alone isn't
enough for the LLM — it lacks a spatial compass to the goal. RL internalizes goal geometry via
trial-and-error; the local vision LLM reasons in text but can't convert that into sustained navigation.
This IS the headline RL-vs-LLM result. Full arc: `docs/superpowers/plans/2026-06-29-llm-agent.md`.

---

## Agent 088 — CORRIDOR FINAL ATTEMPT RL-1 (R1+R4) (2026-07-06, LAUNCHED)

**Hypothesis (findings §2/§4, RL-1 row):** agent_079's recipe (bidirectional Go-Explore frontier +
Violet curriculum anchors + `EXPLORATION_SCALE=4.0`) already connects the corridor archive end-to-end
and pushes the START policy past the Phase-1 wall segment-by-segment (`reach_cherrygrove` 1.0 @28M,
`reach_route30_gate` 1.0 @37M, accelerating) — but then plateaus at Route 30 (`nav/reach_route31` = 0.0
for ~83M, 44M→127M) because `EXPLORATION_SCALE=4.0` also lures the policy into dead-end maps (Dark
Cave/Sprout Tower — 235 cave cells harvested). **R1** generalizes the agent_087 lesson (structural fix
beats reward tweaks: `CONFINE_TO_GYM` took the gym slice 40%→100% by removing the wander-grind OPTION,
not by reward engineering) to the corridor: the new gated `CONFINE_TO_CORRIDOR` flag (env + config,
default off, wired in `train_cnn`/`evaluate_cnn` — commit 6a7fb64) ends the episode the instant the
agent leaves `CORRIDOR_LEGAL`, so entering a cave/tower now costs the *rest of the episode* instead of
paying `+0.4` for nothing. **R4** (frontier archive re-scored by `max_waypoint` ordinal instead of the
egg-quest-era tiers — already landed, commits 25188af + c0dbebb) concentrates ε-greedy frontier resets
at the leading edge of the corridor instead of uniformly across it.

**Config recovery.** `agents/rl/config.py` was in agent_087's gym-slice state (`CONFINE_TO_GYM=True`,
`EXPLORATION_SCALE=0.0`, `FRONTIER_ENABLED=False`, pure 12×`violet_city_gym` curriculum, lr 5e-5, flat
ent_coef 0.005, `TOTAL_TIMESTEPS_CNN=5M`, `CHECKPOINT_FREQ_CNN=1M`, warm from `agent_085@5M`). Config.py's
own git history has no literal `RUN_NAME="agent_079"` commit (the file is edited in place between runs,
rarely committed per-run), so the 079 recipe was reconstructed from this file's inline historical
comments (each field documents its full agent-by-agent chain) cross-checked against training_log's
agent_078/079/083 entries. **Every delta applied, reverting 087's gym state back to the 079 corridor
recipe:**

| Field | 087 (gym, before) | 088 (corridor, after) | Source |
|---|---|---|---|
| `CURRICULUM_STATES_CNN` | 12×`violet_city_gym` | 3×`violet_city` + 2×`violet_city_gym` + 7×`egg_delivered_clean` (4 frontier) | 078 entry (bidirectional curriculum, unchanged into 079) |
| `EXPLORATION_SCALE` | 0.0 | **4.0** | 079 entry ("BOOSTED 1.0→4.0") |
| `CONFINE_TO_GYM` | True | **False** | corridor task must not confine to the gym |
| `CONFINE_TO_CORRIDOR` | False (default) | **True** | R1, the new variable for this attempt |
| `FRONTIER_ENABLED` | False | **True** | 077→079 continuous bidirectional frontier |
| `FRONTIER_SEED_FROM` | None | None (unchanged) | 079 also ran with no seed |
| `LEARNING_RATE_CNN` | 5e-5 (086 stabilize) | **7e-5** | 083 entry: "Hyperparams = 075's gym-stabilize settings (lr 7e-5...)" — 7e-5 was the value carried from 076 through 079 into the gym pivot |
| `ENT_COEF_CNN` (start) | 0.005 flat (086) | **0.02** (anneals to 0.005 over 8M, unchanged `ENT_COEF_CNN_END`/`ENT_ANNEAL_STEPS_CNN`) | same 083 cross-reference |
| `TOTAL_TIMESTEPS_CNN` | 5,000,000 | **60,000,000** | RL-1's own gate (findings §4) — NOT 079's 480M budget (that's R5, later) |
| `CHECKPOINT_FREQ_CNN` | 1,000,000 (082 warm-test) | **5,000,000** | 077-081 corridor-era value; 082's 1M was for an abandoned/unresolved warm test |
| `INIT_FROM_CHECKPOINT` | `agent_085_4999980` | **`agent_079_129999792`** | Task 3 spec — the Route-30-capable checkpoint |
| `env/pokemon_env_cnn.py: MAX_STEPS` | 2**17 (131072, agent_082) | **2**16 (65536)** | agent_082 bumped this as an isolated warm-start test that was never resolved (project pivoted to the gym slice before a result was logged); reverted to 079's actual episode length rather than silently carrying forward an untested variable — outside the brief's stated file list but required by "episode length as 079 used" |
| `EGG_MARKER`, `N_STEPS_CNN`, `BATCH_SIZE_CNN`, `N_EPOCHS_CNN`, `FRONTIER_N_ENVS/P/MAX_STEPS/MAX_CELLS/CELL_K/EPSILON`, `GAMMA`, `GAE_LAMBDA` | — | unchanged | never touched by the gym pivot |

**Kill criterion (findings §4, RL-1 row):** `nav/reach_route31` still 0.0 at 40M → stop (confinement
alone insufficient even warm-started). Extend toward 200–480M (R5) only if it cracks.

**Smoke test (background, ~1 min, killed cleanly via `pkill -f "python -m agents.rl.train_cnn"`):**
CUDA detected (RTX 5080, 16.6 GB), frontier archive enabled at `runs/frontier_archive/agent_088`
(4/12 frontier envs), checkpoint warm-started from `agent_079_129999792_steps.zip`, `learning_rate`
overridden to `7e-05`, `ent_coef` to `0.02`, TensorBoard dir `runs/agent_088_1` created, first two
rollouts logged cleanly (fps ≈2645, `front/reach_route31` 0.867 — the frontier envs already sit past
Route 31 from the warm checkpoint's own archive; `nav/*` all 0 as expected for a single rollout), no
traceback.

**Real launch:** `nohup .venv/bin/python -m agents.rl.train_cnn > runs/agent_088_launch.log 2>&1 &`,
**PID 26490**, launched 2026-07-06 ~14:30. Confirmed alive >60s, visible in `nvidia-smi` as a
`C` (compute) process using ~1 GB VRAM on the RTX 5080. Logs: `runs/agent_088_launch.log`,
`runs/agent_088_2` (TensorBoard), `runs/checkpoints/agent_088/` (every 5M steps).

---

## LLM final attempt (spatial compass arc)

Corridor task (New Bark → Violet Gym badge) from `saves/egg_delivered_clean.state`, per
`docs/superpowers/plans/2026-07-06-llm-spatial-compass.md`. `LLMConfig.state_path` default now points
at the corridor state (the gym slice used `saves/violet_city_gym.state`).

### LLM-1 — calibration (L2 coordinate fix only)

- **Date:** 2026-07-06. **Config:** qwen3-vl:8b (Ollama), corridor start (New Bark, map (24,4)),
  confinement off per the plan (but see caveat below), `max_steps=500`, temperature 0.3, image on.
- **Summary:** steps 500, tokens 1,634,258, battles_won 0, tiles 9, **max_waypoint 0**,
  stopped `max_steps`, wall clock 5,797 s (~97 min, ~11.6 s/step).
  Trace: `runs/llm_logs/run_1783341589.jsonl`.
- **Behavior:** the agent never leaves New Bark Town — all 500 steps on map (24,4), only 9 tiles
  visited. Its thoughts misground the entire episode: the stale gym GOAL text makes it believe it is
  "in the Violet City Gym trying to reach Falkner" while standing in New Bark, so it has no reason to
  look for the town exit at all. By step 11 it drifts to true (7,15) and stays parked there for the
  remaining 489 steps, issuing `move left` 428 times (the tool's "moved left x1" reports presses, not
  displacement — position never changes) plus 58 empty-response fallback `a`-presses; the anti-fixation
  note ("you already tried left — blocked wall") is read in the thoughts but ignored in the action.
  The L2 fix itself works: thoughts quote the true un-swapped coordinates, e.g. "(7,15)".
- **Caveat for LLM-2:** `agents/llm/agent.py` still hardcodes the gym-slice confinement — `home_map`
  locks to the STARTING map, `probe_walkable` filters out any direction that changes the map (exits),
  and map changes are undone. 0 undo events fired this run (the agent never reached an exit), but even
  a perfect navigator could not have left New Bark. This machinery must be disabled/parameterized for
  the corridor before LLM-2, or `max_waypoint` can never exceed 0.
- **Verdict:** navigation fails exactly as the findings doc predicts — this is the calibration
  baseline for L1 (`navigate_to` A* tool).

### Agent 088 — RL-1 verdict (2026-07-06, stopped at 41.3M/60M per kill criterion)

Kill criterion fired: `nav/reach_route31 = 0.0` at 40M with `nav/ep_max_waypoint` FLAT at 2 from
37M→41M (the rising-signal extension condition was not met). What R1+R4 DID buy vs agent_079's
plateau: the start policy consolidated Cherrygrove AND the Route-30 gate (wp 0→2 between 30-37M,
`reach_cherrygrove 1.0`, `reach_route30_gate 1.0`) with corridor confinement active and no off-path
stall — the 079 failure mode (Dark Cave farming) is GONE, confirming the structural hypothesis. The
frontier side reached the gym (`front/ep_max_waypoint 5.0`), so the archive spans the corridor
end-to-end; what's missing is start-policy consolidation past wp 2 — exactly what RL-2's staged
resets + event-scaled episode budget (R2) target. RL-2 will warm from `agent_088_39999936_steps.zip`.

---

## Agent 089 — CORRIDOR FINAL ATTEMPT RL-2 (RL-1 + R2) (2026-07-07, LAUNCH PENDING)

**Hypothesis (findings §2/§4, RL-2 row):** RL-1 proved the structural half — corridor confinement
killed the off-path stall and the start policy consolidated wp 0→2 — but plateaued at the Route-30
gate while the frontier archive already spans the whole corridor (`front/ep_max_waypoint` 5.0). The
gap is start-POLICY CONSOLIDATION lagging the archive, not missing exploration. **R2** attacks that
consolidation lag directly with two structural, additive tricks: **(a) staged resets** — curriculum
env slots reset from the corridor's own on-corridor intermediate saves (all egg-delivered, same story
flags, so the foreign-state segregation that ruled out the 047-051 curricula does not apply), putting
direct gradient AT the lagging Route-30→31→Violet segment every rollout instead of only via the
shared archive; **(b) earned episode budget** (`DYNAMIC_EPISODE_BUDGET=True`, env feature commit
2c6ee45, the Pokémon-Red paper trick arXiv:2502.19920) — start/curriculum episodes begin capped at
16,384 steps and only earn a longer cap (`min(65536, 16384*(1+waypoint_ordinal))`) by reaching a NEW
waypoint, so workers desync and the gradient concentrates on the frontier segment instead of long
tail-wandering after a stall.

**Config deltas vs 088** (everything else identical — CONFINE_TO_CORRIDOR=True, EXPLORATION_SCALE=4.0,
frontier on/4 envs/waypoint-scored, lr 7e-5, ent 0.02→0.005@8M, 60M budget, checkpoint every 5M):

| Field | 088 (RL-1) | 089 (RL-2) |
|---|---|---|
| `RUN_NAME` | agent_088 | **agent_089** |
| `INIT_FROM_CHECKPOINT` | `agent_079_129999792` | **`agent_088_39999936`** (RL-1's kill-point checkpoint: wp 0→2 consolidated) |
| `DYNAMIC_EPISODE_BUDGET` | False (feature not yet landed at launch) | **True** (R2b) |
| `CURRICULUM_STATES_CNN` | 3×violet_city + 2×violet_city_gym + 7×egg_delivered_clean | **2×crossing + 2×route31 + 2×violet_city + 6×egg_delivered_clean** (R2a) |

**Curriculum split math (N_ENVS_CNN=12):** majority 6/12 stays on `egg_delivered_clean.state` (the
true start distribution); the other 6 split evenly (2 each) across the three staged saves past the
wp-2 plateau (`crossing`, `route31`, `violet_city`). Env ranks: 0-1 crossing, 2-3 route31,
4-5 violet_city, 6-11 egg_delivered_clean; with `FRONTIER_N_ENVS=4` the dedicated frontier ranks
(8-11) land inside the egg_delivered_clean block exactly as in 088, ranks 6-7 stay pure-start.
`is_start_env` matches only `egg_delivered_clean`, so the staged slots log under `front/` and `nav/`
remains the clean success gate.

**Kill criteria (findings §4 RL-2 row + tightened early gate):** no new segment
(`nav/reach_violet` == 0.0) after 60M → stop; **also stop early at 30M if `nav/reach_route31` is
still 0.0** (tighter than RL-1's 40M gate — the staged resets sit ON Route 31/Violet already, so a
crack should show much sooner than an archive-mediated hand-off would).

**Smoke test (2026-07-07, ~3 min background boot, GPU shared with the Ollama eval — smoke only, NO
launch):** CUDA detected (RTX 5080, 16.6 GB), frontier archive enabled at
`runs/frontier_archive/agent_089` (4/12 frontier envs), warm-start confirmed from
`agent_088_39999936_steps.zip`, `learning_rate` overridden to `7e-05`, `ent_coef` to `0.02`,
TensorBoard dir `runs/agent_089_1` created, 12 SubprocVecEnv workers up (forkserver), 8 rollouts
logged cleanly (fps ~1190-1373 with Ollama co-resident on the GPU; first rollout already shows
`front/reach_route31` 1.0 from the staged envs, `nav/*` 0 as expected), 185 frontier cells harvested,
no traceback. Killed cleanly via SIGTERM to the exact smoke PID (47641); all 14 child PIDs verified
gone, GPU freed (only the Ollama eval process remains).

**LAUNCH PENDING** (controller launches when the GPU frees up after the LLM-2 runs).
Launch command (controller's step, verbatim from 088's protocol):
`nohup .venv/bin/python -m agents.rl.train_cnn > runs/agent_089_launch.log 2>&1 &`

### LLM-2 — navigate_to tool (L1+L2) — 2026-07-07

**Harness fixes required first (all committed):** the June gym confinement was hardcoded in
`agent.py` (home_map lock + snapshot-undo of map changes + probe exit-filter) and GOAL/SYSTEM_PROMPT
were still gym-slice text → gated behind `confine_to_home_map` (97a05c8). qwen3-vl's thinking-only
empty replies (no off-switch exists for the VL line: think:false and /no_think both ignored) spiked
to 98% on the longer corridor prompt, and the empty-fallback 'a' press self-fed NPC textbook loops
(402 consecutive presses in run 1) → retry-on-empty + overworld wait fallback + slimmer prompt
(0048ec1). Caveat recorded: LLM-1's baseline ran WITH the then-hardcoded lock, so its wp=0 partly
reflects the harness.

**Run 1** (`run_1783362673.jsonl`, pre-0048ec1): 500 steps, wp 0, 11 tiles. navigate_to called ONCE
(worked: "navigated to (6,6)"); then an NPC dialogue loop ate the episode (402× 'a').
**Run 2** (`run_1783379968.jsonl`, post-fix, truncated at 263 steps by the controller — verdict
already unambiguous): empty rate 36% (from 98%), navigate_to 56 calls — real adoption — but **50/56
target the SAME mid-town coordinate (12,11)** which the tool reports unreachable ("navigated to
(13,11)"); it never targets the west exit despite the prompt naming map-edge exits. 9 unique tiles,
map (24,4) only, wp 0.

**Verdict (kill criterion met):** the spatial compass EXECUTES correctly but the model cannot pick
strategic targets with it — local perseveration replaces exploration. Exactly the failure mode the
findings predicted; proceed to LLM-3: the HARNESS owns the leg goals (macro-waypoint checklist,
prompt carries only the current leg's target coordinate for navigate_to).

### Agent 089 — RL-2 verdict (2026-07-07, stopped at 29.3M/60M per the 30M early gate)

Kill criterion fired AND the attempt REGRESSED: `nav/reach_route31` still 0.0 at 29.3M and
`nav/ep_max_waypoint` COLLAPSED 2.0 → 0.0 (agent_088 had consolidated Cherrygrove + the gate; 089
lost both). Post-mortem hypotheses (for RL-3+ design): (a) the 16,384-step base budget truncates
start episodes before they re-earn the consolidated behavior — the budget is earned per NEW waypoint,
so a policy that needs a warm-up stretch gets starved (catastrophic-forgetting pressure instead of
frontier focus); (b) halving pure-start envs (6 of 12 slots to staged saves + frontier absorption)
cut the nav gradient share. The frontier side stayed healthy (front/ep_max_waypoint 4.0). Lesson:
R2's two structural tricks fought R1's consolidation instead of compounding it — the budget quantum
and env split need to be gentler (e.g. base 32k, +16k/waypoint; 8-9 pure-start slots) if retried.
Next per the findings schedule: RL-3 (visited-coords observation, de-transposed, COLD run, full
stack) — the paper's "indispensable" input the 10e ablation removed while it was drawn transposed.
Best corridor checkpoint remains agent_088@40M (wp 2 consolidated).
