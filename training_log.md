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

## PPO_10 — Per-episode route waypoints + gamma=0.995 (2026-05-24)
- 100M steps, 8 envs, ent_coef=0.05, **gamma=0.995**, lr=3e-4
- Curriculum: invariato (2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city)
- **New**: `episode_maps` set, reset ogni episodio — per-episode waypoints:
  - Cherrygrove (26,3): +25/episode
  - Route 31 (26,1): +50/episode
  - Violet City West (26,2): +80/episode
  - Violet City Main (10,5): +100/episode
  - Gym (10,7): +200/episode + rimane +400 one-shot
- Battle win reward (+15) ancora presente da PPO_9 — non rimosso
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+983, avg tiles=101 — battle grinding confermato: agent macina wild battles vicino a New Bark Town invece di navigare
- **Lesson**: battle win reward crea local optima devastante: +15/vittoria near start è più prevedibile dei waypoint distanti. Tiles=101 è la firma del grinding (pochi tile, molti step in battaglia). Rimuovere completamente il battle win reward.

---

## PPO_11 — Removed battle win + gamma=0.999 + ent_coef=0.08 (2026-05-24 → 2026-05-26)
- 100M steps, 8 envs, **ent_coef=0.08** (era 0.05), **gamma=0.999**, lr=3e-4
- Curriculum: invariato (2×start + 2×mid_route30 + 2×before_elm_delivery + 2×violet_city)
- **Rimosso**: battle win reward (commentato in compute_reward)
- Per-episode waypoints invariati da PPO_10
- **Result**: ep_rew_mean=280–284 (stabile, no regressione). reward_events=0.0108 smoothed (solo Elm delivery dai curriculum envs). value_loss=0.0079 (migliore di sempre). entropy=-2.04 stabile per tutta la run. in_battle=0.075 (grinding eliminato). visited_tiles smoothed=155.
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+162.5±43.8, avg tiles=314.3±32.1, avg steps=16384 (tutti troncati, nessun episodio termina). Reward ≈ tiles − step_penalty: nessun waypoint event in quasi tutti gli episodi.
- **Lesson**: infrastruttura perfetta (no collapse, no grinding, value_loss record), ma credit assignment irrisolto. L'agente esplora ~314 tile ma non naviga direzionalmente verso Violet City. I per-episode waypoints (+25/+50/+80/+100) non sono abbastanza forti da trainare la policy a percorrere 800+ step in modo consistente. Serve un curriculum bridge in Route 31.

---

## Save states — aggiornati (2026-05-26)
| File | Posizione | Uso curriculum |
|------|-----------|---------------|
| `start.state` | New Bark Town (24,4) | env 0-1 (PPO_12) |
| `mid_route30.state` | Cherrygrove/Route 30 (26,3), uovo preso | env 2 (PPO_12) |
| `route_31.state` | Route 31 (26,1) | env 3 (PPO_12) — nuovo bridge |
| `before_elm_delivery.state` | Lab Elm (24,5), naming done | env 4-5 (PPO_12) |
| `violet_city.state` | Violet City (10,5), fuori dal PC, squadra curata, 5 pokemon | env 6-7 (PPO_12) |
| `violet_city_gym.state` | Violet City Gym (10,7), 2 passi dentro | rimosso dal curriculum |

---

## PPO_12 — route_31.state bridge curriculum (2026-05-26 → 2026-05-27)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- **Curriculum**: 2×start + 1×mid_route30 + **1×route_31** + 2×before_elm_delivery + 2×violet_city
- route_31.state: salvato manualmente il 2026-05-26 da save_state.py, mappa (26,1) verificata
- Reward invariato rispetto a PPO_11
- **Result**: ep_rew_mean finale 231–238 (declino lieve negli ultimi rollout). reward_events smoothed 0.0095 a fine run (era 0.0571 a 46% — il picco mid-training è dovuto al gym one-shot +400 esaurito, + Elm delivery da curriculum). in_battle smoothed 0.1845 (più alto di PPO_11 a causa del route_31 con encounter rate elevato). visited_tiles smoothed 176. Badge mai ottenuto.
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+62.0±81.0, avg tiles=224.7±85.8. Alta varianza (std±81) suggerisce policy non ancora stabile sulla navigazione dal punto di partenza.
- **Lesson**: tutti i waypoint del percorso sparano (Cherrygrove, Route 31, Violet City Main), ma il badge richiede battere Falkner — e il Pokemon lead è a livello 5. L'agente non ha informazione sul proprio livello nell'obs space, quindi non può valutare se è abbastanza forte per la palestra. Il route_31 bridge migliora il segnale di navigazione (reward_events peak più alto di PPO_11 in mid-training) ma non risolve il battle competence problem.

---

## PPO_13 — Lead level in obs + heal reward + MAX_STEPS×4 (2026-05-27 → 2026-05-28)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- Curriculum: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 2×violet_city)
- **New obs feature**: `lead_level / 100.0` aggiunto all'obs vector (dim 11 → 12). RAM address: **0xDA49** (offset +0x1F da wPartyMon1=0xDA2A). Verificato empiricamente: start.state → lead_level=5 (Totodile lv5). Nota: indirizzo inizialmente impostato a 0xDA4B (byte Unknown nella struct) — corretto a 0xDA49 prima del training.
- **New reward**: heal reward +30 quando `hp_ratio - prev_hp_ratio > 0.4` fuori battaglia → incentiva uso Centro Pokémon prima della palestra.
- **MAX_STEPS**: 2^14 → 2^16 = 65,536 (4× più lungo). N_STEPS: 2048 → 4096 (doubled).
- **Result**: ep_rew_mean finale 62–66 (calo vs PPO_12 ~234 in TensorBoard). reward_events smoothed 0.0028 (quasi azzerato — waypoint e gym quasi mai raggiunti). value_loss stabile. Badge mai raggiunto.
- **Eval da start.state (10 episodi, MAX_STEPS=16,384 ridotto)**: badge=0/10, avg reward=+241.8, avg tiles=246.2.
- **Lesson**: il calo di ep_rew_mean è un artefatto di MAX_STEPS 4×: episodi più lunghi accumulano più step penalty (-0.01 × 65k = -655 max/ep vs -163 prima). L'eval con MAX_STEPS ridotto mostra +241.8 vs PPO_12 +62.0 — l'agente esplora di più. Il vero problema: step penalty -0.01 è eccessiva per episodi da 65k step e schiaccia il reward signal. Cambiata a -0.001 per PPO_14.

---

## PPO_14 — Enemy obs + step penalty -0.001 (2026-05-28 → ...)
- 100M steps, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati)
- Curriculum: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 2×violet_city)
- **Change 1**: step penalty -0.01 → **-0.001**. Penalità max per episodio: 65,536 × 0.001 = 65.5 (era 655).
- **Change 2**: obs dim 12 → **14**. Due nuove feature:
  - `enemy_lead_level / 100.0` — RAM `0xD0FC` (DataCrystal verified). Valore stale per ~100-500 step dopo BATTLE START (RAM non ancora inizializzata), poi stabile per tutta la battaglia. Verificato empiricamente: Falkner Pidgey→7, Pidgeot→9. ✓
  - `enemy_hp_ratio` — `(D0FF/D100) / (D101/D102)`. Scende man mano che si fa danno, torna a 1.0 quando esce il secondo Pokemon. Brevi drop a 0.0 durante animazioni/menu — normale. ✓
- Rationale: l'agente ora può stimare se è in vantaggio o svantaggio in battaglia (confrontando lead_level vs enemy_lead_level e i rispettivi hp_ratio). Segnale diretto per imparare "cura prima di entrare in palestra" e "attacca finché l'avversario ha hp alto".
- **Eval da start.state (10 episodi)**: badge=0/10, avg reward=+299.6±61.3, avg tiles=304.0±58.0, avg steps=16384 (tutti troncati — eval usa MAX_STEPS=2**14=16384, diverso dal training 2**16=65536).
- **Training finale**: ep_rew_mean picco ~498 a 35M steps, poi exploitation collapse a ~386-395 a fine run. reward_events smoothed 0.0028 (identico a PPO_13 — nessun miglioramento). visited_tiles smoothed 243-282 (declino da 360 a metà training). ep_len_mean 34,500-36,000. entropy_loss stabile -2.01/-2.05. explained_variance 0.952-0.997.
- **Lesson**: exploitation collapse: l'agente esplora meno nel tempo — converge a "gironzola vicino alla partenza e sopravvive". Le feature nemico (enemy_lead_level, enemy_hp_ratio) non hanno sbloccato la navigazione perché l'agente non raggiunge il gym abbastanza spesso da usarle. Causa radice identificata: violet_city_gym.state peso 0 → il modello non ha mai allenato il combattimento diretto con Falkner in nessuno dei 14 training run.

---

## PPO_15 — Damage reward + gym curriculum + N_STEPS 8192 (2026-05-29 → 2026-05-30, FERMATO a 58%)
- **500M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4 (invariati), CHECKPOINT_FREQ=25M
- **Curriculum**: 2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + **1×violet_city** (era 2) + **1×violet_city_gym** (era 0, riattivato)
  - Nota sul gym env: il one-shot +400 non spara (mappa già in visited_maps all'init — PPO_8 lesson). Il per-episode +200 spara se l'agente esce e rientra. Beneficio principale: training diretto sui battle con i trainer in palestra e con Falkner → damage reward fire dall'env gym.
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

## PPO_16 — Stuck penalty calibrata (2026-05-30 → 2026-06-02, COMPLETATO)
- **500M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192 (invariati)
- **Curriculum**: invariato da PPO_15
- **Unica modifica**: `stuck_penalty -0.02 → -0.003` (7× ridotto)
- **Training final**: ep_rew_mean **+523** (vs −180 di PPO_15 — calibrazione confermata efficace). visited_tiles smoothed 244-319. policy_gradient_loss −0.0001/−0.001 (policy quasi convergente). explained_variance 0.98+ (value function molto fitted). in_battle smoothed **0.124** (12.4% del tempo in combattimento — sintomo di local optima wild battle). hp_ratio 0.77-0.91. reward_events 0.0046-0.0147 (curriculum-driven).
- **Eval da start.state (10 episodi)**: badge=**0/10**, avg reward=**+180.5±92.0**, avg steps=20,499±17,054, avg tiles=261±82.5. Episodi 4 e 6 raggiungono reward 317-349 ma nessun badge.
- **Gap training/eval**: +523 (training) vs +180 (eval start.state). Le curriculum states `violet_city_gym.state` sparano eventi gratis che gonfiano ep_rew_mean ma la policy non transfer al percorso completo.
- **Failure mode osservato** (gameplay manuale post-training): l'agente entra nell'erba, ingaggia wild Pokémon, perde la battaglia → HP=0 → episodio termina prima del gym. Il `damage_reward` (k=5.0) introdotto in PPO_15 ha creato un attrattore locale: combattere wild = reward immediato, navigare = reward distante.
- **Root cause**: il reward locale del damage in wild battle (+5.0 × delta_hp per turno) compete con il reward distante del gym. PPO greedy → l'agente preferisce l'erba.
- **Lesson**: il damage_reward non distingue tra battaglie utili (gym) e dannose (wild). Va rimosso, e le wild battles fuori dal gym vanno penalizzate esplicitamente per spezzare il local optima.

---

## PPO_17 — Wild battle penalty -3.0 (2026-06-02, FERMATO a 22% / 44.6M step)
- **200M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192
- **Curriculum**: invariato da PPO_16
- **Modifica 1**: Wild battle penalty `−3.0/step` quando `battle_type == 1 AND (map_bank, map_number) != (10,7)`
- **Modifica 2**: Damage reward RIMOSSO completamente
- **Result a 44.6M step (22%)**: FERMATO per stallo oscillante. ep_rew_mean oscilla nel range **[-5000, -2000]** senza trend monotonico (best -1973 a 27M, regressione a -3705 a 44M). `in_battle` oscilla 0.003↔0.083, `visited_tiles` 94↔288, `reward_events` quasi sempre 0.
- **Root cause (calcolo)**: il penalty −3.0/step è strutturalmente troppo aggressivo. Una wild battle media (~80 step pyboy) = −240. Per attraversare Route 30+31 sono fisiologiche 5-10 wild battle inevitabili → costo −1200/−2400 vs +455 totali di waypoint reward dal start a Violet Gym. L'agente fa la matematica corretta: muoversi costa più di stare fermo → equilibrio di stallo.
- **Signal pattern identificato**: "stallo oscillante" ≠ "stagnazione" ≠ "collapse". entropy_loss -2.04 sano, value_loss basso, MA reward oscilla senza convergere. Firma diagnostica: penalty calibrato male rispetto al reward landscape.
- **Lesson**: il wild penalty deve essere abbastanza forte da disincentivare grinding ma abbastanza piccolo da non sopprimere la navigazione. -3.0 è 3× sopra il break-even. Calibrato a -1.0 in PPO_18.

---

## PPO_18 — Wild battle penalty calibrata a -1.0 (2026-06-03, FERMATO a 51% / 103M step)
- **200M steps**, 8 envs, ent_coef=0.08, gamma=0.999, lr=3e-4, N_STEPS=8192 (invariati da PPO_17)
- **Curriculum**: invariato (2×start + 1×mid_route30 + 1×route_31 + 2×before_elm_delivery + 1×violet_city + 1×violet_city_gym)
- **Unica modifica vs PPO_17**: `wild_battle_penalty −3.0 → −1.0` (3× ridotto)
- **Result a 103M step (51%)** — dati last-100 rollouts:
  - `ep_rew_mean` mean=**−1160**, std=166 (range stabile [-1326, -994])
  - `in_battle` mean=**0.0495** (2.5× meglio di PPO_16 a 0.124) ← fix wild penalty funziona
  - `visited_tiles` mean=**214** ± 43 (esplorazione limitata ma stabile)
  - `reward_events` mean=**0.0055** ± 0.009 (sparsi waypoint, non consistenti)
  - `hp_ratio` mean=**0.90** ± 0.06 (sopravvivenza ottima)
  - Best ep_rew_mean: -587 a 49M (mai più raggiunto)
- **Confronto PPO_17 vs PPO_18 a parità di step**:
  - 27M: PPO_17 -1973, PPO_18 -1348 (+32%)
  - 44M: PPO_17 -4184, PPO_18 -1501 (+64%)
  - 50M: PPO_17 -3500, PPO_18 -956 (+73%)
  - Math della calibrazione era corretta, ma non sufficiente.
- **Root cause: "stable suboptimal convergence"**. La policy ha convergato a un equilibrio negativo. Diagnostica:
  - `explained_variance` 0.99+ → value function ha overfittato ai return correnti
  - `policy_gradient_loss` -0.001 → gradiente di policy quasi morto
  - `entropy_loss` -2.04 stabile → la policy non sta più esplorando attivamente
  - `ep_rew_mean` std/mean = 14% → bassa varianza, comportamento ripetitivo
  - Pattern definitivo: non è oscillazione (PPO_17), non è collapse (PPO_2), non è grinding (PPO_9). È **convergence to local optimum**.
- **Lesson finale del filone MLP**: dopo 18 run con tutte le combinazioni di reward shaping (curriculum, milestones, calibration), observation features (lead_level, enemy_obs, party_levels, flag bits), e hyperparameter tuning (gamma, ent_coef, n_steps), il badge da start.state non è mai stato ottenuto in evaluation. Il limite **non è di reward engineering**: lo state vector `(map_id, x, y, hp, levels, ...)` non porta informazione spaziale sufficiente perché PPO costruisca un piano di navigazione di 800+ step verso il gym. CnnPolicy vede il sentiero, l'erba, gli NPC e la porta del gym — informazione che il vettore non contiene.
- **Decisione 2026-06-03**: chiusa la fase MlpPolicy. Switch a **CnnPolicy + frame stacking** (PPO_19+, vedere sezione dedicata).

---

## PPO_CNN_1 — Primo run CNN, stable suboptimal (2026-06-03, FERMATO a 4M / 8%)
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

## PPO_CNN_2 — Curriculum diverso + 8 envs + clip stretto (2026-06-03 → 2026-06-04, CRASH a 95% ma SUCCESS) ⚡

**Primo run in 20 (PPO_1..18 MLP + PPO_CNN_1) a raggiungere ep_rew_mean POSITIVO da start.state curriculum.**

- **50M steps**, **8 envs**, gamma=0.999, gae_lambda=0.95
- **Curriculum** (vs PPO_CNN_1):
  - 3× start.state (main target, 37.5%)
  - 1× mid_route30 + 1× route_31 + 1× before_elm_delivery + 1× violet_city + 1× violet_city_gym
- **Hyperparameter fix vs PPO_CNN_1**:
  - `LEARNING_RATE_CNN` 2.5e-4 → **1.5e-4** (più conservativo)
  - `BATCH_SIZE_CNN` 256 → **512** (sfrutta GPU)
  - `clip_range` 0.2 → **0.1** (update meno aggressivi)
- **Result a 47.4M (95%, CRASH per bug GIF path)**:
  - `ep_rew_mean` BEST: **+112.68** at step 41.5M
  - `ep_rew_mean` Last-100: **+53.6 ± 18.9** (PRIMO POSITIVO STABILE in 20 run)
  - `visited_tiles` Last-100: 296 ± 47 (vs PPO_18 a 214) — esplora 40% più ampio
  - `in_battle` Last-100: 0.045 (vs PPO_18 a 0.05) — evita wild battle
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
- **Checkpoint salvati**: `runs/checkpoints/PPO_CNN_2/PPO_CNN_2_20000000_steps.zip` (al picco intermedio +26) e `PPO_CNN_2_40000000_steps.zip` (vicino al picco massimo +68).
- **Lesson**:
  1. Il **curriculum** + **clip range stretto** + **batch più grande** = combo vincente. Il PPO ha imparato per la prima volta a navigare in modo significativo.
  2. `clip_fraction` resta alto (0.34) anche con clip_range=0.1, ma il training rimane stabile — sintomo che la policy continua a evolversi attivamente, non collasso.
  3. `reward_events` sempre basso (0.002) MA `ep_rew_mean` positivo significa che il guadagno viene da **exploration tiles** + **eventi rari** + risparmio penalty. La policy NON ha ancora "scoperto" Violet City Gym da start.state.
- **Next step**: evaluation da `start.state` su 10 episodi del checkpoint 40M. Se badge > 0 → success. Se badge = 0 → PPO_CNN_3 con focus su credit assignment del segmento finale (start → gym).

---

## PPO_CNN_3 — Damage reward al gym + heal reward (2026-06-04, COMPLETATO)

**Eval da start.state: badge=0/10. L'agente impara a esplorare ma non combatte mai (in_battle=0%).**

- **50M steps**, 8 envs, lr=1.5e-4, n_steps=2048, batch=512, n_epochs=4, ent_coef=0.03, clip_range=0.1
- **Curriculum**: invariato da PPO_CNN_2 (3×start + 1×mid_route30 + 1×route_31 + 1×before_elm_delivery + 1×violet_city + 1×violet_city_gym)
- **Modifica vs PPO_CNN_2**: damage reward RE-INTRODOTTO ma **map-constrained al gym** (10,7): `(prev_enemy_hp_ratio - enemy_hp_ratio) * 10.0` solo se `battle_type > 0` per t e t-1 E mappa == (10,7). Evita il local optima wild di PPO_16 (lì il damage era globale).
- **Result training**: ep_rew_mean picco ~+250 a ~32M step, poi declino lento a fine run. visited_tiles smoothed ~466 (esplorazione molto ampia — record CNN). `in_battle` smoothed ~0.0 → l'agente evita sistematicamente ogni combattimento.
- **Eval da start.state (10 episodi, 3 checkpoint)**: badge=**0/10** su tutti. avg reward ~+175/+188, **in_battle = 0% in OGNI episodio**.
- **Gameplay osservato (--watch)**: l'agente NON si muove da New Bark Town. Esplora le case, rientra in casa propria, sale/scende i piani, ma **non entra mai nell'erba** (obbligatoria per raggiungere Cherrygrove). Non è un bug di in_battle=0: è una policy che massimizza i tile reward dentro New Bark senza mai partire.
- **Root cause**: il reward landscape penalizza l'uscita. New Bark Town + interni offrono ~+250 di tile reward "gratis" e sicuri; uscire verso Route 29 → wild battle penalty (-0.05/step) + nessun tile nuovo immediato che batta i +250 già accumulati. Il damage reward al gym è irraggiungibile da start.state: l'agente non arriva mai al gym, quindi quel segnale non propaga.
- **Lesson**: il damage reward map-constrained al gym è inerte da start.state (credit assignment troppo lungo). Il vero problema è che la navigazione iniziale (uscire da New Bark) non è mai incentivata abbastanza forte da superare l'attrattore locale dell'esplorazione interna. Serve un **mega-bonus per waypoint** che renda raggiungere Cherrygrove/Route 31/Violet enormemente più redditizio del gironzolare. → PPO_CNN_4.

---

## PPO_CNN_4 — Mega-bonus waypoint per-episodio (2026-06-04 → 2026-06-05, COMPLETATO 100M)

**Eval da start.state: badge=0/10. L'agente ORA combatte (in_battle 3.2%) ma il comportamento è "schizofrenico" tra due policy in conflitto — peggiore in eval dei CNN precedenti.**

- **100M steps** (override manuale, config dice 100M), 8 envs, hyperparameters invariati da PPO_CNN_3
- **Curriculum**: invariato
- **Modifica vs PPO_CNN_3**: per-episode waypoints potenziati **~80×** per spezzare l'attrattore "resta a New Bark":
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
- **Confronto con CNN_2/CNN_3**: CNN_4 in eval è **PEGGIORE** (+89 vs +188/+196) nonostante il training salga a +626. La std scende da 86 (40M) a 53 (60M): la policy si consolida su un equilibrio mediocre "combatti un po', esplora moderato, mai vince", abbandonando l'exploration policy stabile e ad alto reward di CNN_3.
- **Root cause: policy segregation visiva (limite strutturale, NON di reward shaping)**. La CnnPolicy impara policy separate per aspetto visivo: gli env curriculum (che partono già avanti) imparano a combattere e raccolgono i mega-bonus → ep_rew_mean training sale; gli env start.state imparano a esplorare. Le due policy **non transferiscono** perché la rete le associa a scene visive diverse. Il mega-bonus 80× ha cambiato il comportamento nei curriculum env (in_battle ora >0) ma ha destabilizzato la policy start.state (explained_variance crolla a 0.25). Pattern di declino post-peak identico a CNN_3.
- **Lesson finale del filone reward-shaping CNN**: dopo CNN_2/3/4, tre run consecutivi 0/10 badge da start.state con lo stesso pattern strutturale (training sale, eval non riflette, policy segregation). Il reward shaping ha raggiunto il suo limite: aumentare i waypoint cambia COSA fanno i curriculum env, non fa GENERALIZZARE la policy a start.state. Il segnale di reward denso non basta — manca un segnale di **novelty intrinseco e visivo** che spinga la policy start.state a uscire e continuare a esplorare territorio nuovo a prescindere dai waypoint hardcoded.
- **Decisione 2026-06-05**: pivot a **KNN visual novelty** (approccio Whidden / PokemonRedExperiments). → PPO_CNN_5.
  - **CORRECTION 2026-06-09**: after inspecting the reference repo, Whidden's **V2** (the version that
    reaches Cerulean = past the 1st gym) **replaced KNN with coordinate-based exploration** — which this
    project already has (`visited_tiles` keyed on `(map_bank, map_number, local_x, local_y)`). KNN is the
    *abandoned* V1. So PPO_CNN_5 does **not** add KNN; it realigns the existing setup to the V2 recipe.

---

## PPO_CNN_5 — Whidden V2 realignment: reward rescale + pure single-start (2026-06-09, RECIPE LOGGED, RUN PENDING)

**Hypothesis**: the blocker across CNN_2..4 was not representation (env is already CNN + frame-stack like
PokemonRedExperiments) nor "reward-shaping exhaustion", but five concrete divergences from Whidden V2:
inverted reward scale, heterogeneous curriculum (policy segregation), no level/opponent reward, no
self-state in obs, over-long episodes. CNN_5 fixes scale + curriculum + level reward + episode length
first (cheap validation); self-state in obs and battle competence are deferred to Phase 4.

- **30M steps (SHORT VALIDATION)**, **12 envs**, lr=1.5e-4, n_steps=2048, batch=512, n_epochs=4,
  ent_coef=0.03, clip_range=0.1, gamma=0.999, gae_lambda=0.95, VecFrameStack(4). CHECKPOINT_FREQ=5M.
- **Curriculum**: PURE single start — **12× start.state** (was 3×start + 5 mixed). Removes the
  visually-keyed sub-policies that never transferred to start.state in eval (the CNN_4 segregation cause).
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
    CNN_2 at 95%).
  - eval badge detection: `evaluate_cnn` now reads `infos[0]["zephyr"]` (terminal-step info), not a
    post-auto-reset RAM read — the old code returned `badge=no` even on a win.
  - op_level stale-RAM bug: enemy level (`0xD0FC`) is garbage outside battle; reward now gated on
    `battle_type > 0` + legal-range clamp.
- **New instrumentation**: per-EPISODE navigation metric `nav/reach_{cherrygrove,route31,violet_west,
  violet_main,gym}` (fraction of episodes reaching each waypoint) + `nav/ep_max_waypoint`. Logged on
  episode end (NOT step-averaged), so it is un-confounded by dwell time. This is the honest success signal
  — `ep_rew_mean` misled CNN_2..4 (training rose while eval stayed 0/10).
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
  = 60M > 30M total → **no intermediate checkpoints saved, only `PPO_CNN_5_final.zip`**.
- **Lesson**: scale realignment + single-start solved escape + segment-1 navigation, but a flat per-tile exploration
  reward cannot cross distant chokepoints. → **PPO_CNN_6**: hybrid exploration (small episode-new +0.005 to keep the
  corridor warm + dominant **lifetime-new +0.02** for frontier expansion), **warm-started from `PPO_CNN_5_final`**,
  with the checkpoint-frequency bug fixed (save_freq divided by N_ENVS).

---

## PPO_CNN_6 — Lifetime tile-novelty (warm-start from CNN_5), STOPPED ~55% (2026-06-09)

**Pure single-start + lifetime novelty → reward death-spiral; frontier still walled at Route 31.**

- 30M target (stopped ~16.7M), 12× start.state, warm-started from `PPO_CNN_5_final`.
  Reward: episode-new **+0.005** + lifetime-new **+0.02** (×0.1).
- **Result**: `nav/reach_violet_west` = **0.000** throughout; `ep_max_waypoint` capped at 2.0 (same wall as CNN_5).
  `ep_rew_mean` **declined monotonically +2.59 → +0.38**; `visited_tiles` 570 → 380; `ep_len_mean` 32768 → 28565.
- **Root cause**: lifetime novelty SATURATES. Once the known corridor is "seen forever", re-tread pays only the
  tiny +0.005 trail reward and the lifetime bonus never fires at the *undiscovered* frontier — so the dense corridor
  reward that kept CNN_5 busy was stripped away with nothing to replace it. The agent contracted into a smaller,
  lower-reward routine. Novelty only pushes a frontier the agent OCCASIONALLY crosses; it can't CREATE discovery.
- **Bonus**: confirmed the checkpoint-freq fix (intermediate checkpoints at 3/6/9/12/15M saved).
- **Lesson**: reward-shaping (episode vs lifetime novelty) cannot solve a DISCOVERY problem. Reverted in CNN_7.

---

## PPO_CNN_7 — Small curriculum + realigned reward (warm-start), COMPLETED 30M (2026-06-09 → 2026-06-10)

**Curriculum made the start-state policy CATASTROPHICALLY FORGET segment-1 navigation — exposed only by the new start-state-filtered nav metric. `ep_rew_mean` ~1.5 hid the regression completely.**

- 30M, 12 envs = **8× start + 2× route_31 (=Violet West 26,2) + 2× violet_city (10,5)**, warm-started from `PPO_CNN_5_final`.
- Reward reverted to CNN_5 episode-novelty (**+0.02/new tile**, ×0.1). `nav/reach_*` filtered to start.state episodes
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
  the same structural tension: CNN_5 (stall — can't discover), CNN_6 (death-spiral — novelty saturates),
  CNN_7 (forgetting — curriculum segregates). → architectural pivot needed: **frontier state-sharing**
  (Go-Explore-lite), where reset states are the start policy's OWN trajectory edge — continuous with start-state
  experience, so no high-reward island and no segregation.

---

## PPO_CNN_8 — Egg-quest STORY GATE + reverse curriculum (2026-06-09 → 2026-06-10, IN PROGRESS)

**THE big discovery (user's gameplay knowledge): the "Route 31 wall" of CNN_5/6/7 is a STORY GATE, not a
navigation chokepoint.** Two trainers block Route 30's WEST branch until the Mystery Egg is delivered to Elm
(confirmed: Bulbapedia + gameplay). The agent never did the egg quest (`reward_events`=0) → permanently
blocked. This OVERTURNS the CNN_5/6/7 "discovery" diagnosis. (The CNN_7 "frontier state-sharing" plan was
abandoned the moment the gate was found — state-sharing can't open a scripted gate.)

**Map constants RE-VERIFIED 2026-06-10 by walking the route (instrumented save_state.py). Old labels WRONG:**
(26,3)=Cherrygrove · (26,1)=Route 30 north / GATE zone · (26,2)=Route 31 POST-GATE (Dark Cave area) ·
(26,11)=Route31↔Violet gatehouse · (10,5)=Violet · (10,7)=Gym · (26,10)=Mr.Pokemon's house · (3,70)=Dark Cave.
Old code had `ROUTE_31`=(26,1) and `VIOLET_CITY_WEST`=(26,2) — both wrong; fixed. Route 30 forks: WEST→Route 31
(gated), EAST→Mr.Pokemon's house (dead-end).

**Approach: REVERSE CURRICULUM** — single start-state per run (→ no segregation), warm-start chain. Egg events
weighted up: `egg_received`+3, `egg_delivered`+5 (opens gate); Mr.Pokemon house +2 gated on "egg not yet received".

- **Stage 1** (`egg_delivered`, warm CNN_5): reached (26,1) gate zone but did NOT cross to (26,2) even with the
  gate OPEN. Causes: (a) a BUG — Mr.Pokemon house +2 fired post-egg, luring the agent to the EAST dead-end off
  the correct WEST path; (b) warm-start aversion. `ep_max_waypoint`=2 flat.
- **Stage 1b** (Mr.Pokemon reward gated on egg-not-received): agent now TRIED the west path — but DIED in the
  battle gauntlet. `ep_len_mean` crashed 32768→17500 (early termination = blackouts), `in_battle` 0.5, `hp` 0.6.
  `reach_route31` max 0.33, never consolidated. → **BATTLE COMPETENCE is a wall on the ROUTE itself**, not just at the gym.
- **Crossing stage** (`crossing.state` = (26,1) doorstep PAST the west trainers, Totodile lv11): reliably crosses
  to Route 31 (`reach_route31`=1.0), SURVIVES (`ep_len`=32768, `in_battle`~0.05, `hp`~0.98), explores Route 31
  (~700 tiles) — but stalls at Route 31 (26,2) → Violet (10,5): `reach_violet`=0 flat. New chokepoint = the
  GATEHOUSE building door (26,11).

**Pattern / lessons (CNN_8):**
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

## PPO_CNN_8_crossing_wp — Waypoint rewards from crossing.state (2026-06-10 21:36 → 06-11 00:05, STOPPED 27.4M/30M)

Warm variant of the crossing stage WITH the new per-episode latched waypoint rewards (+2 each).

- **Waypoint rewards WORK as a door gradient**: `reach_route31`=1.0 stable, `ep_max_waypoint`=3,
  `ep_rew_mean` 1.73 (peak 2.16), full survival (`ep_len`=32768, `hp_ratio`=1.0).
- **But**: `reach_violet`=0 flat for the whole run — the Route31→Violet GATEHOUSE door (26,11) was never
  crossed despite its +2 waypoint waiting. The +2 (0.2 post-scale) is apparently too small vs ~700 tiles
  (1.4 post-scale) of comfortable Route 31 exploration income.
- **in_battle → 0.004**: TOTAL battle avoidance learned. No reward pays for fighting; battles only cost HP.

## PPO_CNN_9_selfstate — First Dict-obs run, cold start (2026-06-11 00:08 → 02:22, STOPPED 23.8M/30M)

First run with the new Dict observation (image + 7-float self-state vector) + MultiInputPolicy, cold start
(old CnnPolicy checkpoints incompatible). Log: `runs/cnn9_train.log`.

- Nav metrics PEAKED mid-run (`reach_route31` max 1.0, `ep_max_waypoint` max 3) then **collapsed to 0** by
  the end. `ep_len_mean` fell 32768 → 22.7k (early terminations = blackouts), `in_battle` 0.097, final
  `ep_rew_mean` 1.32 (peak 1.70).
- **Pattern: battle-incompetence death spiral** — the agent pushes the frontier, gets killed by trainers/wilds,
  the policy regresses to safe wandering. Same wall as stage1b, now visible end-to-end in one run.

## PPO_CNN_9_gymtest — Phase-A battle test INSIDE the gym (2026-06-11 02:44 → 03:37, STOPPED 10.3M/30M) ⚠️ KEYSTONE

Cold start from `violet_city_gym.state` (2 steps inside Falkner's gym): pure battle-competence test, no grass.

- **The agent WALKS OUT of the gym and tours Violet City**: visited_tiles 614–866 (the gym alone has ~100),
  `in_battle` peaked 0.63 early then decayed to 0.11, `ep_len` ~31.7k ≈ always truncation, **badge never won**,
  `ep_rew_mean` 2.80 (earned by sightseeing, not fighting).
- **Direct reward arithmetic confirms it is RATIONAL**: touring Violet ≈ 600 tiles × 0.02 = 12 pre-scale,
  vs full gym fight chain = gym damage 6 + badge 10 = 16 pre-scale but discounted by risk of death and battle
  length — exploration income structurally dominates combat income. The same inequality explains ALL the
  avoidance/grinding failure modes of the last 30 runs.
- **Conclusion**: this is not a representation or curriculum problem anymore — it is a reward-geometry problem.
  → full redesign (PPO_CNN_10) instead of further mini-patches.

---

## PPO_CNN_10 — REDESIGN: event-dominant reward + story/combat obs + 150M single run (2026-06-11, PLANNED)

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

**Ruled out by this redesign analysis**: resuming PPO_CNN_8_stage1 with higher ent_coef (east-fork bias is in
the weights; stage1b already was that experiment post-bugfix and hit the battle wall) · catch/evolution rewards
(not needed for Falkner — scope creep) · RecurrentPPO / PufferLib port (no failure traces to the optimizer).

---

## Filone CnnPolicy — Inizia da PPO_19 (2026-06-03)

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

## Strategies To Try (After PPO_12)

### If PPO_12 succeeds (badge obtained from start.state):
- Valutazione quantitativa: badge rate su 100 episodi da start.state
- Passare all'agente LLM (Phase 5 del plan) per il confronto

### If PPO_12 fails (route_31 envs raggiungono gym ma start.state no):
- Aggiungere un secondo bridge (es. violet_city_west.state o un punto su Route 29)
- Oppure aumentare ulteriormente i per-episode waypoints per i primi segmenti del percorso

### Ruled out / do not retry
- Revisited tile penalty (PPO_2 lesson)
- visited_maps reset per episode (PPO_3 lesson)
- Map transition bonus > +30 per-lifetime (causes reward hacking — PPO_3 lesson; per-episode è ok se piccolo)
- Unverified Sprout Tower flags 0xD85C/0xD85D (wrong addresses)
- ent_coef < 0.05 with this reward scale (premature convergence)
- Map Card event reward (not in standard flag range, Cherrygrove +20 suffices)
- sprout_tower_2f in curriculum (isola narrativa, non contribuisce al path verso Violet City)
- 3×before_elm_delivery senza VecNormalize (troppa varianza di return, causa regressione — PPO_6 lesson)
- violet_city_gym.state nel curriculum SOLO per il milestone one-shot +400 (mappa già in visited_maps all'init — PPO_8 lesson). Riadottato in PPO_15 con peso 1 per battle training diretto — il one-shot non spara, ma damage reward e badge reward sì.
- Battle reward proporzionale a HP (catena causale troppo lunga, doppia lettura RAM nello stesso step sempre 0)
- Battle win reward flat +15 (crea local optima: grinding near start.state > navigare verso waypoint distanti — PPO_10 lesson)
- **Stuck penalty -0.02** (PPO_15 lesson) — penalty troppo aggressivo: 28,740 step × 0.02 = −575/ep vs +260 new tile reward. Ratio 2.2:1 contro l'esplorazione. Il training si blocca a −180 di ep_rew_mean da 130M step in poi. Usare −0.003 (breakeven < 0.009).
- **Damage reward in wild battles** (PPO_16 lesson) — `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0` indistinto tra wild/trainer/gym crea local optima: l'agente combatte wild Pokémon (reward immediato +5×delta per turno) invece di navigare verso il gym (reward distante). Risultato: `in_battle` smoothed = 0.124, badge=0/10 in eval. Rimosso in PPO_17. Se mai re-introdotto, deve essere map-constrained al gym (10,7).
- **Wild battle penalty -3.0/step** (PPO_17 lesson, fermato a 22%) — overshoot opposto del damage_reward: penalty troppo aggressivo crea stallo oscillante. 1 wild battle media (80 step) = −240; per attraversare Route 30+31 con 5-10 wild battle inevitabili: −1,200/−2,400 vs +455 waypoint reward → netto negativo per muoversi → agente preferisce stare fermo. Calibrato a -1.0 in PPO_18.
- **MlpPolicy + state vector** (filone PPO_1→PPO_18 chiuso 2026-06-03) — 18 run con tutte le combinazioni di reward shaping, curriculum, observation features. Mai badge da start.state in eval. Convergenza a "stable suboptimal" in PPO_18 con `explained_variance` 0.99+ e `policy_gradient_loss` ~0. Lo state vector non porta informazione spaziale sufficiente per piani di navigazione lunghi. Filone abbandonato a favore di CnnPolicy.
- **CnnPolicy senza curriculum diversity** (PPO_CNN_1 lesson, 2026-06-03) — 4×start.state convergeva a stable suboptimal a -500 in 4M step. Il problema non era solo rappresentazionale ma anche di propagazione del reward: il policy network ha bisogno di esempi "facili" dai curriculum envs per imparare associazioni stato→azione che generalizzano. Confermato che 8 envs misti + clip stretto + batch grande sblocca il break-even (PPO_CNN_2).
- **Damage reward map-constrained al gym da solo** (PPO_CNN_3 lesson) — inerte da start.state: l'agente non raggiunge mai il gym, quindi il segnale non propaga. In più non risolve l'attrattore "resta a New Bark" → in_battle=0% in eval. Va abbinato a un incentivo forte all'uscita.
- **Mega-bonus waypoint per-episodio (80×)** (PPO_CNN_4 lesson) — aumentare i waypoint a +300/+500 fa salire ep_rew_mean in training (i curriculum env li raccolgono) ma NON fa generalizzare la policy a start.state, e destabilizza il value (explained_variance 0.25). Eval peggiore di CNN_2/CNN_3. Sintomo di **policy segregation visiva**: la CnnPolicy impara policy distinte per aspetto visivo che non transferiscono. Limite strutturale del reward shaping con curriculum eterogeneo.
- **Reward shaping per risolvere policy segregation** (filone PPO_CNN_2→4 chiuso 2026-06-05) — tre run consecutivi 0/10 badge da start.state. Calibrare i reward (wild penalty, damage, waypoint) cambia il comportamento dei curriculum env ma non fa generalizzare a start.state. Serve un segnale di **novelty intrinseco visivo** (KNN frame embedding) invece di waypoint hardcoded. → filone KNN (PPO_CNN_5).

---

## Future Upgrades — Ispirate a PokemonRedExperiments

Backlog ordinato per impatto stimato. Da provare in questo ordine se PPO_13+ non raggiunge il badge da start.state in modo consistente.

### Già implementate
- [x] Lead Pokemon level in obs (PPO_13) — 0xDA49
- [x] Heal reward (PPO_13) — +30 quando hp_ratio aumenta >0.4 fuori battaglia
- [x] MAX_STEPS 4× (PPO_13) — 2^16 = 65,536
- [x] Step penalty ridotta (PPO_14) — -0.01 → -0.001
- [x] Opponent level in obs (PPO_14) — 0xD0FC (DataCrystal verified, Falkner Pidgey=7 / Pidgeot=9 ✓)
- [x] Enemy HP ratio in obs (PPO_14) — 0xD0FF/D100 current, 0xD101/D102 max
- [x] N_STEPS 8192 (PPO_15) — 4096 → 8192. Un episodio da 65k step copre ~8 rollout invece di ~16 → meno errori di bootstrap accumulati per episodio.
- [x] Damage reward (PPO_15) — `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0`. ✓ formula corretta.
- [x] violet_city_gym.state nel curriculum (PPO_15, peso 0→1) — per training diretto su Falkner. One-shot +400 non spara (mappa già in visited_maps), per-episode +200 spara su re-entry.
- [x] Stuck penalty (PPO_15 → PPO_16) — introdotto a -0.02 in PPO_15, calibrato a **-0.003** in PPO_16. Con -0.02 il penalty (575/ep) dominava l'exploration reward (260/ep) → training stagnante a −180. Breakeven: penalty < 0.009/step. -0.003 porta il baseline a +315/ep (positivo). Diverso da PPO_2 (quello era -0.01/step flat sempre, senza distinzione new/old tile).
- [x] Event flags binari in obs (PPO_15) — 3 feature binarie agli indici 8-10: rival beaten, egg received, egg delivered. Obs dim 13→15 (sostituisce i 2 byte normalizzati).
- [x] Party levels tutti e 6 (PPO_15, dim 15→20) — indici 15-19: slot 2-6. Struct size=0x30 calcolato, addresses: 0xDA79/0xDAA9/0xDAD9/0xDB09/0xDB39. Verificare con test_enemy_level.py.
- [x] Gym battle exit reward +150 (PPO_15) — map-constrained (10,7). Fires su battle falling edge dentro gym. Max 3×/ep. Safe vs grinding (lesson PPO_10).
- [x] Wild battle penalty (PPO_17 → PPO_18) — introdotto a `−3.0/step` in PPO_17, troppo aggressivo (stallo oscillante a 22%). Calibrato a `−1.0/step` in PPO_18. Map-constrained: NOT (10,7). Math: 1 wild battle ≈ 80 step → costo −80 (era −240) → gradiente verso il gym diventa marginalmente positivo (+455 waypoint − 600 wild penalty ≈ −145 netto, vs −1,345 di PPO_17).
- [x] Damage reward rimosso (PPO_17) — eliminato `(prev_enemy_hp_ratio - enemy_hp_ratio) * 5.0`. Era responsabile dell'attrattore wild battle.

### Livello successivo (se PPO_15 fallisce)
| Upgrade | Stato | Descrizione | Impatto stimato | Costo |
|---------|-------|-------------|-----------------|-------|
| **DataCrystal event tracking completo** | 🔲 da fare | Reward per ogni event flag da New Bark Town a Falkner. Gym trainer 1/2 flags da scoprire con `test_enemy_level.py start`. Nota: running shoes non esiste in Gen 2, Pokédex/Pokéball già coperti da ELM_BIT. | Alto | Medio (ricerca + verifica empirica) |
| **Image-based observation** | 🔲 ultimo resort | Switch da MlpPolicy + vettore a CnnPolicy + screen (72×80 RGB) come PokemonRed. L'agente vede il gioco come un umano. | Molto alto | Molto alto — refactoring completo, 10× più compute |
