# Holos Medyk — Training Data Generation Plan

## Goal
Use Claude Code subagents to generate thousands of diverse, specific prompts in two categories. These prompts will later be sent to DeepSeek R1 and MedGemma 27B to produce training responses. This plan covers **prompt generation only** — no API calls, no scripts.

---

## Prompt Set A: DeepSeek R1 — Warzone Trauma Prompts (~2K prompts)

These are open-ended prompts simulating what a real Ukrainian civilian would say to a voice assistant during/after a bombardment. They should feel raw, panicked, specific, and varied. NOT templated — each one should read like a unique person in a unique situation.

### What makes a good DeepSeek prompt
- Feels like a real person talking, not a test question
- Contains situational details (where they are, what happened, who's hurt)
- Varies wildly in specificity — some people describe everything, some just scream "there's blood"
- Includes emotional state implicitly (rambling = panic, terse = shock, detailed = calm-ish)
- Covers the full spectrum: single injury, multiple casualties, self-injury, children, elderly, trapped

### Topic coverage (ensure spread across all of these)
- Hemorrhage / bleeding (limbs, torso, head, embedded objects)
- Airway / breathing problems (dust inhalation, choking, unconscious person)
- Crush injuries (building collapse, trapped limbs, entrapment)
- Burns (thermal, chemical, electrical)
- Blast injuries (shrapnel, glass, concussive, ear damage)
- Shock (recognition, what to do)
- Fractures / dislocations (improvised splinting)
- Chest wounds (open, closed, breathing difficulty)
- Head / spinal injury (when NOT to move someone)
- Triage (multiple casualties, who to help first)
- Safety/refusal scenarios ("should I pull out the metal?", "what pills should I give?", "should I do surgery?")
- Psychological crisis (panic attacks, children screaming, frozen/unable to act)
- Environmental hazards (gas leak, structural instability, fire, flooding)

### Language split
- ~1,500 in English
- ~500 in Ukrainian (colloquial, not formal — "скрізь кров" not "геморагія")

### Subagent structure
Spawn 4 subagents in parallel, each generating ~500 unique prompts:
- **Subagent 1**: Hemorrhage, blast injury, embedded objects, wound care (~500 prompts)
- **Subagent 2**: Crush injury, fractures, spinal, entrapment, structural collapse (~500 prompts)
- **Subagent 3**: Burns, airway, shock, chest wounds, chemical/gas exposure (~500 prompts)
- **Subagent 4**: Triage, safety/refusal, psychological crisis, environmental hazards, Ukrainian-language prompts (~500 prompts)

Each subagent writes its output to a separate file:
- `data/training/prompts/deepseek_batch_1_hemorrhage_blast.txt`
- `data/training/prompts/deepseek_batch_2_crush_fracture.txt`
- `data/training/prompts/deepseek_batch_3_burns_airway_shock.txt`
- `data/training/prompts/deepseek_batch_4_triage_safety_ukrainian.txt`

One prompt per line. No numbering, no metadata, just the raw prompt text.

---

## Prompt Set B: MedGemma 27B — General Medical Prompts (~1.5K prompts)

These are medical knowledge questions that a civilian might ask. Broader than trauma — covering general emergency medicine reasoning, symptom assessment, severity evaluation, when to worry vs when not to. NOT imaging, NOT radiology, NOT pathology, NOT dermatology.

### What makes a good MedGemma prompt
- Focused on clinical reasoning, not visual/imaging tasks
- Tests the model's ability to explain medical concepts simply
- Includes "is this serious?" type questions
- Covers medication safety (not dosing — but "is it safe to take X?", "what are signs of allergic reaction?")
- Includes infection recognition, wound care basics, chronic condition emergencies (diabetes, asthma, seizures)

### Topic coverage
- Symptom assessment ("my child has X, Y, Z — how bad is this?")
- Severity classification ("when do I need to go to a hospital vs handle at home?")
- Basic pharmacology safety (allergic reactions, contraindications, NOT dosing)
- Infection recognition (wound infection, sepsis warning signs)
- Chronic condition emergencies (asthma attack, diabetic crisis, seizure, allergic anaphylaxis)
- Pregnancy / childbirth emergencies (basic only)
- Pediatric vs adult differences
- Hypothermia / hyperthermia
- Dehydration and fluid management
- Pain management (non-pharmacological)
- When NOT to intervene / common dangerous myths
- Mental health crisis (suicidal ideation, acute psychosis — model should know limits)

### Language split
- ~1,200 in English
- ~300 in Ukrainian

### Subagent structure
Spawn 3 subagents in parallel, each generating ~500 unique prompts:
- **Subagent 5**: Symptom assessment, severity classification, infection, wound care (~500 prompts)
- **Subagent 6**: Chronic emergencies, pediatric, pregnancy, hypothermia/hyperthermia (~500 prompts)
- **Subagent 7**: Pharmacology safety, myths/misconceptions, mental health, Ukrainian-language medical prompts (~500 prompts)

Output files:
- `data/training/prompts/medgemma_batch_1_symptoms_severity.txt`
- `data/training/prompts/medgemma_batch_2_chronic_pediatric.txt`
- `data/training/prompts/medgemma_batch_3_pharma_myths_ukrainian.txt`

One prompt per line.

---

## Execution

In the new chat:
1. Spawn subagents 1-7 in parallel
2. Each subagent generates ~500 diverse, specific, non-templated prompts and writes them to its output file
3. That's it. Prompts only. API calls come later.

## Quality bar
- Every prompt should be unique — no two should describe the same scenario
- Prompts should vary in length (5 words to 3 sentences)
- Prompts should vary in emotional register (calm, panicked, clinical, broken, rambling)
- No template artifacts ("{{TOPIC}}", "[INSERT]", etc.)
- Ukrainian prompts should use colloquial civilian language, not medical terminology
