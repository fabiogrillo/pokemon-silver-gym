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
- violet_city_gym.state nel curriculum (gym milestone non può sparare per quell'env — mappa già in visited_maps all'init — PPO_8 lesson)
- Battle reward proporzionale a HP (catena causale troppo lunga, doppia lettura RAM nello stesso step sempre 0)
- Battle win reward flat +15 (crea local optima: grinding near start.state > navigare verso waypoint distanti — PPO_10 lesson)

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

### Livello successivo (se PPO_14 fallisce)
| Upgrade | Descrizione | Impatto stimato | Costo |
|---------|-------------|-----------------|-------|
| **Party levels tutti e 6** | Leggere livello di tutti i party pokemon (offset +0x1F dai successivi wPartyMon, ogni struct = 48 bytes). Segnale più ricco per battle readiness. | Medio | Basso |
| **Stuck penalty leggero** | -0.02 per ogni tile rivisitata nella sessione (non per episodio — usa visited_tiles set). Scoraggia cicli locali senza penalizzare l'esplorazione globale. Diverso da PPO_2 (che era -0.01/step puro). | Medio | Basso |
| **Event flags in obs** | Aggiungere flag_rival e flag_elm come features binarie nell'obs (già letti dal RAM reader). L'agente può sapere se ha già consegnato l'uovo o battuto il rivale. | Medio | Bassissimo — già in RAM reader |
| **N_STEPS aumentato** | 2048 → 8192. Rollout più lunghi = value function vede più di ogni episodio per rollout. Con MAX_STEPS=65536 e N_STEPS=2048, un episodio dura ~32 rollout — troppi bootstrap. | Medio | Medio (più RAM, training più lento per update) |
| **Image-based observation** | Switch da MlpPolicy + vettore a CnnPolicy + screen (72×80 RGB) come PokemonRed. L'agente vede il gioco come un umano — navigazione e battle molto più intuitive. | Molto alto | Molto alto — richiede refactoring completo, 10× più compute |
| **200M+ timesteps** | PokemonRed usa 5B steps. Noi usiamo 100M. Anche solo 200-300M potrebbe sbloccare comportamenti che non emergono a 100M. | Incerto | Alto (ore di training) |
