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
