# Holos Medyk — Clinical Research Findings
## Based on Trauma Surgeon Advisor Input (Bret, April 5 2026)

This document consolidates deep research triggered by Bret's clinical guidance. It informs the training data strategy, prompt generation, golden dataset curation, and voice assistant behavior design.

---

## 1. MARCH Not ABC — The Framework Must Change

**Bret said ABC. The evidence says MARCH is better for our use case.**

Every major trauma body has shifted to putting catastrophic hemorrhage control **first**. The British military call it `<C>ABC`, ATLS (2023) endorsed x-ABC, and TCCC/TECC use **MARCH** (Massive hemorrhage → Airway → Respiration → Circulation → Head/Hypothermia).

Why: in penetrating/blast trauma (87-96% of Ukrainian civilian injuries), an arterial bleed kills in 3-5 minutes — faster than an airway obstruction. Doing an airway check on someone bleeding from the femoral artery is doing ABC in the order that kills people.

| Protocol | Who Uses It | Ordering |
|---|---|---|
| ABC (classic) | Legacy BLS, basic civilian first aid | Airway → Breathing → Circulation |
| `<C>`ABC / x-ABC | UK military, ATLS 2023+ | Catastrophic hemorrhage → A → B → C |
| **MARCH** | **TCCC, TECC, Ukrainian military/civilian training (FAST/PULSE)** | **Massive hemorrhage → Airway → Respiration → Circulation → Hypothermia** |

Ukrainian NGOs (FAST, PULSE) have already trained ~60,000 civilians on MARCH. Using ABC would contradict what trained Ukrainians already know.

**Decision: Build the voice assistant around MARCH.** Bret's ABC framework maps perfectly — we just reorder so hemorrhage comes first, and add hypothermia as an explicit step. The clinical content Bret provided for each letter is unchanged.

**Sources:**
- x-ABC vs ABC shifting paradigms (PMC, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12094115/
- TCCC Handbook v5 (US Army): https://api.army.mil/e2/c/downloads/2023/01/19/31e03488/17-13-tactical-casualty-combat-care-handbook-v5-may-17-distro-a.pdf
- NAEMSP 2024 poster (410 Medical): https://410medical.com/app/uploads/2024/01/NAEMSP-2024-Poster-Presentation.pdf

---

## 2. Hemorrhage Control — The #1 Lifesaver

Bret identified arterial bleed control as one of the two most important field skills. Research confirms: **untrained laypersons succeed at tourniquet application only 20-22% of the time on their own, but 80.5% succeed with audio/phone instructions from a trained guide.** This is exactly our voice AI use case.

### Hierarchy
1. **Direct pressure** — two hands, full body weight, sustained 5-10+ minutes without peeking
2. **Tourniquet** — for limb bleeding. High and tight, above the wound, not on a joint. Tighten until bleeding **stops completely**. Pain is expected and is NOT a reason to loosen. Mark the time. Do not remove.
3. **Wound packing** — for junctional wounds (groin, axilla, neck) where tourniquet won't work. Push gauze/cloth DEEP into the wound, pack tight, maintain pressure 3+ minutes
4. **Pressure dressing** on top of packing

### Critical myths to kill
- "Tourniquets cause amputation" — modern data: limb loss from properly applied tourniquets is **extremely rare**; safe for 6+ hours. Delay kills.
- Improvised tourniquets without a windlass have a **99% failure rate**. Belt alone almost never works. Must have cloth + windlass (stick, pen, spoon).
- Tampons for wound packing — **debunked by every authority**. A tampon absorbs 9mL; arterial bleed pumps 100x that per minute. Use torn clothing packed deep.
- "Don't peek" is critical — civilians instinctively lift pressure to check if bleeding stopped. This disrupts clot formation.

### What NOT to pack
- Chest wounds (use occlusive seal)
- Abdominal evisceration (cover with moist cloth, don't push organs in)
- Skull wounds with visible brain matter
- Eye wounds

### Hemostatic agents in Ukraine
Celox and QuikClot Combat Gauze are available through military retailers (Abrams.com.ua) and humanitarian distribution. Celox works even on anticoagulated blood.

**Stop the Bleed has Ukrainian-language materials**: https://stopthebleedcoalition.org/for-ukraine/ — 20,000+ Ukrainians trained.

**Sources:**
- Stop the Bleed (ACS): https://stopthebleed.usuhs.edu/
- Safety of civilian tourniquets, 105 cases (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC5104170/
- Trauma System News — 8 pitfalls: https://trauma-news.com/2017/09/stop-bleed-8-pitfalls-avoid-hemorrhage-control/
- JEMS wound packing essentials: https://www.jems.com/patient-care/emergency-trauma-care/wound-packing-essentials-for-emts-and-paramedics/

---

## 3. Pulse-to-BP Heuristic — Valid Ordinal Tool, Wrong Numbers

Bret's heuristic: carotid ≥ 60, femoral ≥ 70, radial ≥ 90 mmHg systolic.

### The Deakin & Low challenge
Deakin CD, Low JL. BMJ 2000;321:673-674. Tested 20 patients in cardiac cath lab with arterial lines:
- Radial pulse lost at mean **78.7 mmHg** (range 52-102), not 80/90
- Femoral pulse lost at mean **56.8 mmHg** (range 42-70), not 70
- Carotid pulse lost at mean **51.3 mmHg** (range 40-62), not 60

The specific mmHg numbers were **removed from ATLS** by the 7th edition (2004). They no longer appear in the 9th (2012) or 10th (2018) editions.

### But the rank-ordering IS valid
Radial is always lost first, carotid always last. This is basic cardiovascular physiology — progressive hypotension shunts blood centrally. McManus et al. (J Trauma 2015) reaffirmed ordinal pulse ranking as a valid triage tool.

| Finding | Interpretation |
|---|---|
| Radial pulse present, strong | Likely not in severe shock |
| Radial absent, carotid present | Significant hypotension, hemorrhagic shock |
| Carotid absent | Cardiac arrest or near-arrest |

### Better civilian shock assessment for the voice assistant
A 4-question binary protocol requiring no equipment:

1. **Consciousness**: "Are their eyes open? Are they responding when you talk to them?" (AVPU proxy)
2. **Skin**: "Touch their hand or arm. Is the skin cold, wet, or clammy?" (earliest shock sign)
3. **Radial pulse**: "Put two fingers on the inside of their wrist, thumb side. Can you feel a heartbeat?" (binary yes/no only — no counting)
4. **Breathing**: "Watch their chest. Are they breathing? Is it very fast or very slow?"

**Decision: Include the pulse check as a binary (present/absent), but do NOT quote specific mmHg numbers.** Frame as: "If you can feel a pulse at the wrist, that's a good sign. If you can only feel it in the neck, that's serious. If you can't feel any pulse, start CPR."

Also include skin signs and mental status as primary shock indicators — they're earlier, easier, and more reliable for untrained civilians.

**Sources:**
- Deakin & Low, BMJ 2000: https://www.bmj.com/content/321/7262/673
- ANDROMEDA-SHOCK (CRT validation), JAMA 2019;321(7):654-664
- McNarry & Goldhill (AVPU validation), BJA 2004;93(3):362-367

---

## 4. Tension Pneumothorax — Include With Strong Gate Conditions

Bret identified this as the second most important combat field skill: diagnosis + needle decompression at 2nd intercostal space, midclavicular line, above the 3rd rib.

### Civilian recognition WITHOUT stethoscope
- Chest wound + rapidly worsening breathing
- One side of chest NOT rising with breaths (visual asymmetry)
- Tracheal deviation (windpipe shifts away from injured side) — run finger down front of throat
- Distended neck veins (JVD) — bulging veins both sides of neck
- Subcutaneous emphysema — "bubble wrap" crackling when pressing skin near wound/neck
- Cyanosis (blue/grey lips and fingertips)
- Rapid deterioration: talking → confused → unresponsive within minutes

### Anatomical site — current consensus
- **2nd ICS, MCL** (Bret's recommendation): easier to talk through verbally. Find midpoint of collarbone, two finger-widths below, insert above 3rd rib.
- **5th ICS, AAL** (alternative): chest wall is thinner here, higher success rate, but harder landmark to describe to a civilian.
- TCCC now accepts both sites. Default to 2nd ICS MCL for voice guidance simplicity; fall back to 5th ICS AAL if first attempt fails.

### The needle length problem
- Standard 3.25cm (1.25") needle fails in **up to 50%** of patients at 2nd ICS MCL
- TCCC minimum: **14G, 8cm (3.25") catheter**
- Even 8cm fails in ~10-15% at 2nd ICS MCL (chest wall too thick)
- At 5th ICS AAL, 8cm succeeds in **>95%** of patients

### The honest question: can a civilian do this?
**Only with adequate equipment.** Without at least a 14G+ catheter of 8cm+ length, needle decompression will likely fail and could cause harm. A sewing needle, pen tube, or short hypodermic won't work.

### Decision tree for the voice assistant
```
Chest wound + rapidly deteriorating →
  Confirm: asymmetric chest rise, JVD, tracheal shift →
    Confirm: no medevac available →
      Confirm: 14G+ catheter, 8cm+ length available →
        YES → Guide decompression at 2nd ICS MCL
        NO adequate supplies → Seal wound, burp if worsening, position injured-side-down, wait
```

### Ethical defensibility
Untreated tension pneumothorax is near-100% fatal within minutes. Ukrainian law (Civil Code Art. 1166) protects Good Samaritan rescuers. The AI is relaying established emergency protocols, analogous to a 911 dispatcher guiding CPR. **Imperfect intervention beats no intervention when mortality is otherwise certain.**

**Sources:**
- Inaba K et al., Arch Surg 2012;147(9):813-818
- TCCC Guidelines 2021/2024 (CoTCCC): https://jts.health.mil/
- Laan DV et al., J Trauma 2016;80(2):272-277

---

## 5. Airway Management — What Civilians Can Actually Do

Bret mentioned Guedel, i-gel, bag-mask, oxygen — but acknowledged these are for medics with equipment.

### Recovery position (unconscious, breathing, no spinal suspicion)
1. Straighten legs. Nearest arm out at right angle, palm up.
2. Far arm across chest, back of hand against near cheek. Hold it there.
3. Pull far knee up, foot stays flat.
4. Roll toward you onto their side using the knee.
5. Tilt head slightly back, mouth angled down for drainage.

### HAINES position (unconscious, breathing, suspected spinal injury)
Less cervical motion than standard recovery. Raise nearest arm overhead past ear, roll onto side resting head on the raised arm as pillow. Use when trauma is suspected and you must leave them.

### Head-tilt-chin-lift vs jaw-thrust
- **Head-tilt-chin-lift**: simple, effective, default for laypeople. Contraindicated if C-spine injury suspected.
- **Jaw-thrust**: preserves spine, but harder for laypeople. Fingers behind jaw angles, push forward/up.
- **AHA/ERC layperson guidance**: use head-tilt-chin-lift regardless, because a dead patient from airway obstruction is worse than a potentially worsened spine.
- **For the voice assistant**: default to head-tilt-chin-lift. If obvious head/neck trauma AND rescuer can maintain position, offer jaw-thrust. Always include the override: "If they still aren't breathing, tilt the head back. Air is more important than the neck."

### Sucking chest wound (open pneumothorax)
**Three-sided seal is OBSOLETE.** TCCC removed that recommendation in 2008. Current guidance:
1. Expose wound, wipe skin dry around it
2. Cover with anything airtight: plastic bag, cling wrap, credit card, duct tape
3. Seal ALL FOUR sides
4. Check for exit wound on back — seal it too
5. **If they get worse after sealing** (tension developing): "burp" the seal — lift one edge, let air escape, reseal

### Inhalation injury — a time bomb
Burns to face, singed nose hairs, soot in mouth, hoarse voice, stridor = airway swelling that can kill in 30 minutes. A civilian **cannot** fix a closing airway. The voice assistant should say: "You cannot open a swollen airway with your hands. Keep them upright, keep them calm, and get them to a medic as fast as possible."

### Rescue breathing — YES in warzone context
Compression-only CPR is the default lay rescuer protocol for cardiac arrest. But **many warzone arrests are hypoxic** (blast, smoke, crush, blood loss, drowning in flooded basements). The assistant SHOULD offer rescue breathing when the rescuer is willing. Also for all pediatric arrests and drowning.

**Sources:**
- TCCC Guidelines 2024: https://learning-media.allogy.com/api/v1/pdf/f4cf1d4e-3191-443a-befc-415838fb04f2/contents
- Crisis Medicine sucking chest wound: https://www.crisis-medicine.com/a-sucking-and-blowing-chest-wound-is-the-sound-of-not-dying/
- AHA 2024 drowning update (Circulation): https://www.ahajournals.org/doi/10.1161/CIR.0000000000001274

---

## 6. Ukrainian Civilian Injury Epidemiology

### Blast dominates everything
- **87-96% of war-related injuries are blast mechanism** (shrapnel, missiles, mines, drones)
- Polytrauma is the norm: 45.5%+ have multiple body regions injured simultaneously
- A single "broken arm" scenario is rare; "shrapnel in thigh + facial lacerations from glass + tympanic rupture + concussion" is typical

### Weapon-specific patterns
- **FPV/Shahed drones**: now #1 civilian killer. 395 killed, 2,635 injured (Feb 2022 – Apr 2025). 72% near frontlines. Small warheads = survivable if bleeding controlled quickly.
- **FAB-500/1500 glide bombs**: ~200-500 kg TNT equivalent. Concussion/TBI near-universal among survivors. Massively underdiagnosed.
- **Cluster munitions**: 987 killed/wounded in 2022 alone, 95% civilians. Dud submunitions act as landmines for years.
- **Thermobaric (TOS-1A)**: 3rd/4th-degree burns, superheated air respiratory damage, barotrauma with less fragmentation.

### Where civilians get hurt
- **Apartments during strikes** (glass, collapse, fire, gas) — #1 location
- **Basements/shelters** (crush from collapse, CO from generators, hypothermia, dehydration)
- **Streets** (drones and cluster submunitions, especially frontline towns)
- **Energy infrastructure** (blast + electrical burns during repair)

### Time to medical care
- Peacetime Ukraine: ~8.5 min urban ambulance response
- **Wartime during mass attacks: EMS effectively doesn't come.** Civilians self-evacuate. Double-tap strikes specifically target rescue workers.
- Rural/frontline: casualties wait hours to days. Civilian vehicles = primary evacuation.
- **The assistant must assume 15 minutes to several hours before any professional help**, especially during drone waves or east of the Dnipro.

### Preventable deaths where layperson intervention saves lives
1. **Exsanguination from extremity wounds** — #1 preventable death. Tourniquet/packing/pressure.
2. **Tension pneumothorax** from chest shrapnel — occlusive seal + decompression if equipped
3. **Airway obstruction** — unconscious blast victims, recovery position
4. **Hypothermia** — Ukraine winters (-10 to -25°C), unheated damaged buildings
5. **Smoke inhalation / CO poisoning** from apartment fires and shelter generators
6. **Crush syndrome** — rhabdomyolysis on extraction from rubble. Hydrate before release if possible.
7. **Blast lung** — delayed presentation, apparently stable patient dying hours later. Must flag for evacuation.

### Civilian training baseline
~60,000 trained out of ~35 million population = **<0.2% have real first aid training**. Assume a frightened layperson with shaking hands, possibly injured themselves, in the dark, with no kit. Improvised tourniquets, pressure with clothing, doors as stretchers.

**Sources:**
- Sirko et al., J Neurotrauma 2025: https://journals.sagepub.com/doi/full/10.1177/08977151251365558
- War-related maxillofacial injuries Ukraine (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12061781/
- Pediatric Kharkiv study (Conflict and Health 2025): https://conflictandhealth.biomedcentral.com/articles/10.1186/s13031-025-00694-w
- OHCHR drones report: https://ukraine.ohchr.org/en/Short-range-drone-attacks-killed-395-civilians-injured-2635-between-February-2022-and-April-2025
- Prehospital lessons from Ukraine (Military Medicine 2024): https://academic.oup.com/milmed/article/189/1-2/17/7255885
- CNA "We Need a Medic" 2024: https://www.cna.org/analyses/2024/11/we-need-a-medic

---

## 7. Chemical Casualties Handbook (Army MMCC 4th Ed.)

Bret's army doctor friend recommended this. **It is NOT a general trauma handbook — it is the US Army Medical Research Institute of Chemical Defense handbook for CBRN chemical casualties.** Actually more relevant than expected given the Russian CBRN threat.

### Nerve agents (highest priority for Ukraine)
- Recognition: miosis (pinpoint pupils), runny nose, chest tightness (mild); sudden unconsciousness, convulsions, apnea, copious secretions (severe)
- Self-aid: 1 MARK I Kit (atropine 2mg + pralidoxime 600mg IM)
- Severe: 3 MARK I Kits + diazepam 10mg IM
- Differential from cyanide: nerve agent = pinpoint pupils + copious secretions + fasciculations; cyanide = normal/dilated pupils + few secretions

### Cyanide (from burning plastics in bombed buildings)
- Recognition: rapid breathing → convulsions → respiratory arrest in 2-4 minutes
- Key clue: severe respiratory distress in a NON-BLUE patient (cherry-red skin possible)
- Even WITHOUT antidotes, vigorous supportive care (ventilation, oxygen) can save patients. Do NOT triage as expectant just because no antidotes available.

### Lung-damaging agents (chlorine from bombed industrial sites)
- Phosgene: smells like freshly mown hay. Heavier than air — pools in basements. **Delayed pulmonary edema — patient feels fine for hours then dies.** Physical exertion dramatically worsens it.
- Chlorine: sharp irritating odor, immediate airway irritation
- **Critical civilian guidance**: if exposed, even if feeling fine, DO NOT physically exert yourself. Rest and observation for minimum 6-8 hours.

### Fentanyl derivatives (Russia used aerosolized carfentanil in 2002 Moscow theater siege)
- Recognition: rapid unconsciousness, pinpoint pupils, respiratory depression
- Antidote: naloxone (Narcan) 0.4-2mg IV/IM, repeat every 2-3 min
- **Naloxone is short-acting** — patient may re-sedate. Monitor 24 hours.
- Airway management is priority: position on side, clear vomitus.

### Decontamination (most civilian-applicable section)
- **Within 1-2 minutes is critical**, especially for mustard
- Best civilian decontaminants (proven equivalent to military kits):
  - Copious water
  - Soap and water (fat-based soap, NOT detergent)
  - Flour followed by wet tissue wipes
  - 0.5% bleach (9 parts water : 1 part bleach) — only if water limited
- **Strip clothing first** — traps vapor and liquid agent
- **Never bleach in eyes or wounds** — water/saline only
- Physical removal > chemical neutralization. Wipe, then wash.

**Full PDF saved to**: `data/references/mmcc-hbk_4th-ed.pdf`

---

## 8. Training Scenarios (Evidence-Based)

Based on the injury epidemiology, these should be priority scenarios for the golden dataset:

**S1 — Apartment strike** (highest frequency): Family in Soviet-era apartment, missile/drone impact nearby. Windows blown, dust, gas leak risk. Child with facial glass lacerations, grandmother with femur crush under furniture, father with confusion/tinnitus (mTBI/blast lung), mother panicking. Triage, improvised bleeding control, decide who to move.

**S2 — FPV drone strike on car, rural road**: Driver alone, drone hit engine, penetrating abdominal fragments + burned hands. No cell signal. Self-aid: abdominal pressure, don't drink water, stay warm, crawl to cover (second drone risk).

**S3 — Metro/shelter during wave**: Elderly chest pain (MI, not trauma), pregnant woman early labor, child asthma attack (inhaler empty), CO symptoms near generator. Non-trauma medical emergencies under resource constraints.

**S4 — School/playground strike**: Multiple pediatric casualties, teacher as caller. Mass casualty triage, tourniquet sizing for children, psychological first aid.

**S5 — Double-tap on rescue scene**: Initial strike, civilians helping, second missile 10 minutes later targeting rescuers. Must teach 5-minute scene delay / cover discipline.

**S6 — Cluster submunition / UXO**: Child picks up submunition, hand amputation + face/chest fragmentation. Scene safety (more submunitions in yard), improvised tourniquet for pediatric arm.

**S7 — Chemical exposure after industrial site strike**: Chlorine/phosgene from bombed facility. Symptom recognition, decontamination, enforced rest, delayed pulmonary edema warning.

**S8 — Winter hypothermia compound**: Any hemorrhage scenario + sub-zero temperatures + no heating. Must cover warm ground insulation and passive warming — hypothermia → coagulopathy → death is the most undertaught lethal pathway.

---

## 9. Key Design Decisions for the Voice Assistant

| Decision | Rationale |
|---|---|
| **Use MARCH, not ABC** | Matches Ukrainian civilian training (FAST/PULSE), matches injury epidemiology (hemorrhage kills first) |
| **Hemorrhage before airway** | Arterial bleed kills in 3-5 min; airway obstruction takes longer in trauma |
| **Include tension pneumothorax decompression** | Only with equipment gate (14G+, 8cm+). Without supplies, guide to seal + position + wait. |
| **Pulse check as binary only** | Present/absent, no mmHg numbers. Ordinal ranking valid, exact numbers debunked. |
| **4-question shock assessment** | Consciousness → skin → radial pulse → breathing. Binary yes/no, no equipment. |
| **Three-sided seal is banned** | Obsolete since TCCC 2008. Fully occlusive + burp if deteriorating. |
| **Rescue breathing offered** | Warzone arrests are predominantly hypoxic, not cardiac. Compression-only is insufficient. |
| **Hypothermia in every hemorrhage scenario** | Even in summer. Blood loss → thermal failure → coagulopathy = lethal triad. |
| **Scene safety with double-tap warning** | 5-10 min delay before approaching fresh impact sites. |
| **Assume no EMS for 30+ min** | Design for prolonged field care, not bridge-to-ambulance. |
| **Assume <0.2% civilian training** | Default to improvised tools, zero jargon, explain tourniquet windlass from first principles. |
| **CBRN recognition included** | Russia has confirmed chemical capability. Civilians near industrial sites need decon guidance. |

---

## 10. Reference Texts (from Bret + Army Doctor Friend)

| Title | Publisher | Relevance |
|---|---|---|
| *ABC of Prehospital Emergency Medicine* (2nd ed.) | Wiley | Primary prehospital textbook |
| *ABC of Transfer and Retrieval Medicine* | Wiley | Patient transport/retrieval |
| *Oxford Handbook of Retrieval Medicine* (Evans et al.) | Oxford | Co-authored by Bret's senior colleague at ARV |
| *Military Medical Care Handbook* (4th ed.) | US Army / globalsecurity.org | CBRN chemical casualties — PDF saved locally |
| TCCC Handbook v5 | US Army | Field trauma protocol gold standard |
| TCCC Guidelines 2024 (CoTCCC) | JTS | Latest clinical practice guidelines |
| Stop the Bleed materials (Ukrainian) | ACS | Civilian hemorrhage control training |
