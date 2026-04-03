# Голос Медик — Holos Medyk
## An Offline, Voice-Interactive Emergency Medical Assistant for Ukrainian Civilians Under Bombardment
### Gemma 4 Good Hackathon — Project Plan v4 (FINAL)

---

## 1. THE PITCH (30 seconds)

Download Holos Medyk. Four gigabytes. Then put your phone in airplane mode.

Say in Ukrainian: "My daughter is bleeding from her arm, there's glass everywhere, I can't stop it."

Gemma 4 talks you through saving her life. Step by step. In Ukrainian. No internet. No cloud. No subscription.

For shelters and aid stations where phones aren't available — a $75 standalone device does the same thing.

**Holos Medyk. A voice when no one else can answer.**

---

## 2. COMPETITIVE LANDSCAPE

**Nothing like this exists.** Exhaustive research confirms:

| Existing Solution | What It Does | Why It's Not Enough |
|---|---|---|
| Pi-Card, Local-Voice (GitHub) | Generic offline voice assistants on Pi | Not medical, not Ukrainian, not fine-tuned |
| Prepper survival LLMs | Uncensored Mistral on Pi for survivalists | Generic, not medically trained, English only |
| AidSnap (app) | AI first aid with camera + voice | Requires internet — useless when infrastructure destroyed |
| Red Cross / IFRC First Aid apps | Static offline first aid guides | Not conversational, not voice-interactive, not context-aware |
| Private LLM SurviveV3 / UltraMedical | Offline survival + medical models on iOS | Wilderness/generic, not warzone-specific, not Ukrainian |
| MamayLM (INSAIT) | First Ukrainian LLM based on Gemma 3 | General purpose, not medical, not voice, not edge-deployed |
| Infermedica (Ukraine refugees) | AI triage for Ukrainian refugees in Poland | Cloud-based, requires internet, text-only |

**Our unique combination that no one has built:**
- LLM fine-tuned on trauma surgery protocols for warzone medicine ✓
- In Ukrainian ✓
- Voice in / voice out ✓
- Runs on any modern smartphone — free ✓
- Also deployable as $75 standalone device for shelters ✓
- Completely offline ✓
- Using Gemma 4 E4B (launched April 2, 2026) ✓

---

## 3. PRIZE TARGETS

| Track | Prize | Fit |
|-------|-------|-----|
| **Main Track** | $50K / $25K / $15K / $10K | Novel, emotional, technically deep |
| **Global Resilience** (Impact) | $10K | Perfect — offline disaster response |
| **Cactus** (Technology) | $10K | "Best local-first mobile application" |
| **LiteRT** (Technology) | $10K | Running E4B via LiteRT-LM on Android + Pi |
| **Unsloth** (Technology) | $10K | Fine-tuning E4B on emergency medical data |
| **Safety & Trust** (Impact) | $10K | Grounded medical info, hallucination prevention |

**Maximum potential: $85K** (Main 1st + Resilience + Cactus + LiteRT + Unsloth + Safety)

---

## 4. TWO-TIER PRODUCT STRATEGY

### Tier 1: Free Android App (PRIMARY PRODUCT)
- For the 20+ million Ukrainians with smartphones
- Gemma 4 E4B running via LiteRT-LM on-device
- ~4GB download, then fully offline forever
- Works on any modern Android phone (6-8GB+ RAM)
- Phone's built-in mic and speaker — no extra hardware
- **Cost to user: $0**

### Tier 2: Standalone Shelter Device (SECONDARY / DEMO)
- For community shelters, hospitals, aid stations, elderly, children
- For situations where phones are destroyed or unavailable
- Raspberry Pi 5 (8GB) running Gemma 4 E4B via LiteRT-LM
- USB mic + 3.5mm speaker + battery bank
- **Prototype cost: $311 (our build)**
- **At-scale BOM estimate: $50-75 per unit**
  - ARM SBC with 4-8GB RAM (~$30-50 at volume)
  - SD card (~$5)
  - USB mic (~$3)
  - Small speaker (~$2)
  - Battery pack (~$10)
- Pitch: "An NGO could deploy 1,000 of these across Ukrainian shelters for the cost of a single armored vehicle."

### Why Both?
- App is the real product — free, instant, already in your pocket
- Pi device covers edge cases: destroyed phones, elderly without smartphones, shared shelter devices
- Both run the same model, same fine-tuning, same voice pipeline
- Pi also serves as development platform and physical demo prop for video

---

## 5. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED CORE PIPELINE                      │
│                                                             │
│   Voice Input ──> Gemma 4 E4B (LiteRT-LM) ──> Piper TTS   │
│       │            - Native ASR (audio in)        │         │
│       │            - Medical reasoning            │         │
│       │            - Ukrainian response           │         │
│       │            - Function calling (RAG)       │         │
│       │                                           ▼         │
│   Microphone                                   Speaker      │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │  LOCAL KNOWLEDGE BASE (embedded in app/device)   │      │
│   │  - Trauma surgeon protocols (brother)            │      │
│   │  - Blast injury: tourniquet, hemorrhage, burns   │      │
│   │  - Crush injuries, structural safety             │      │
│   │  - Chemical/gas exposure                         │      │
│   │  - Triage decision trees                         │      │
│   │  - Shelter locations (cached per city)            │      │
│   │  - Psychological first aid                       │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐        ┌──────────────────────────┐
│   TIER 1: ANDROID   │        │   TIER 2: PI DEVICE      │
│                     │        │                          │
│  Any modern phone   │        │  Pi 5 (8GB) Aluminum     │
│  6-8GB+ RAM         │        │  DUNGZDUZ USB Mic        │
│  Built-in mic/spkr  │        │  WAONIQ 3.5mm Speaker    │
│  ~4GB download      │        │  JUOVI 65W Battery Bank  │
│  FREE               │        │  ~$75 at scale           │
└─────────────────────┘        └──────────────────────────┘
```

---

## 6. MODEL SELECTION

| Model | Effective Params | RAM (4-bit) | Phone? | Pi 5 (8GB)? | Our Choice? |
|-------|-----------------|-------------|--------|-------------|-------------|
| **E2B** | 2.3B | ~1.5GB | ✅ any phone | ✅ | Fallback if E4B too heavy |
| **E4B** | 4.5B | ~5GB | ✅ flagship | ✅ | ✅ **PRIMARY** |
| 26B A4B | 3.8B active | ~18GB | ❌ | ❌ (needs 18GB) | Too large for edge |
| 31B | 31B | ~20GB | ❌ | ❌ | Workstation only |

**E4B is the sweet spot.** Best model that runs on both phones and Pi. Native audio input (ASR), 128K context, function calling, 140+ languages. Fine-tunable via Unsloth.

If E4B is too heavy for older Ukrainian phones, we ship E2B as a "lite" mode. Same app, smaller model, works on phones with 4GB RAM.

---

## 7. EXACT HARDWARE (ORDERED — for Pi dev/demo)

| Item | Cost | Status |
|------|------|--------|
| CanaKit Raspberry Pi 5 Starter Kit PRO - Aluminum (8GB), 128GB SD | $254.95 | Ordered — arriving Sunday Apr 6 |
| DUNGZDUZ USB Microphone (USB-A, plug-and-play) | incl. in $56.29 | Ordered — arriving tonight Apr 3 |
| JUOVI 65W 20000mAh USB-C PD Power Bank (4 ports) | incl. in $56.29 | Ordered — arriving tonight Apr 3 |
| WAONIQ Mini 3.5mm Portable Speaker | incl. in $254.95 | Ordered — arriving Sunday Apr 6 |
| **TOTAL** | **$311.24** | |

CanaKit includes: Pi 5 board (8GB), aluminum case, Samsung 128GB SD (Pi OS preloaded), 5A USB-C PD power supply, USB card reader, 2x micro HDMI cables.

Additional needed:
| Item | Est. Cost | When |
|------|-----------|------|
| GPIO push button + jumper wires | ~$5 | Week 2 |
| Android test phone (if needed) | Kevin's existing phone | — |

---

## 8. TEAM & ADVISORS

### Kevin Power — Builder
- 3x MIT Engineering (SCM '25, TPP '27, EECS '27)
- Winner, OpenAI gpt-oss-20b Red-Teaming Challenge (Kaggle)
- 1st Place, Hack-Nation Global AI Hackathon
- IEEE published (2x) — LLM applications
- Founder @ Powerlab — agentic AI systems
- Experience: Unsloth fine-tuning, Ollama, agentic tool-calling, LLM safety

### Brother — Medical Advisor (Trauma Surgeon)
- Practicing trauma surgeon
- Provides clinical protocols, triage decision trees, medical validation
- Reviews fine-tuning data for clinical accuracy
- Red-teams medical outputs for dangerous advice
- **Key question**: "If you could teach a civilian 20 things to save the most lives after a missile strike, what would they be?"

### Girlfriend — Ukrainian Language & Cultural Advisor
- Native Ukrainian speaker
- Validates translations and medical terminology
- Records speech samples for ASR testing
- Tests device UX and TTS naturalness
- Connects with family in Ukraine for real-world feedback

---

## 9. FINE-TUNING STRATEGY: BILINGUAL APPROACH

### Why English Training → Ukrainian Inference Works

Gemma 4 E4B is natively trained on 140+ languages including English and Ukrainian. The model already "knows" Ukrainian — what it doesn't know is emergency warzone medicine. We teach the medicine in English (from brother), and the model's multilingual capability handles Ukrainian output.

```
FINE-TUNING DATASET COMPOSITION:

~80% English (brother's trauma protocols)
├── Hemorrhage control & tourniquet application
├── Airway management for untrained civilians
├── When to move someone vs NOT move them
├── Crush injury protocol (building collapses)
├── Burns treatment (do's and don'ts)
├── Shock recognition & management
├── Triage decision trees
├── Common mistakes civilians make
└── "NEVER do this" danger warnings

~20% Ukrainian (girlfriend-validated)
├── Medical terminology glossary (~100 key terms)
├── Emergency dialogue examples (20-30 conversations)
├── Ukrainian civil defense documents
└── Common civilian phrasing ("there's blood everywhere" not "hemorrhage")
```

### What to Ask Brother (Priority Order)

**Priority 1 — Life-threatening:**
- Hemorrhage control: direct pressure, tourniquet, wound packing
- Airway management for untrained people
- When to move vs NOT move someone
- Crush injury protocol
- Burns: what to do and NOT do
- Shock recognition and management
- Chest wounds basics
- Triage decision trees

**Priority 2 — Common injuries:**
- Glass/shrapnel wounds (do NOT remove embedded objects)
- Fracture stabilization / improvised splinting
- Head injury assessment
- Eye injuries from debris
- Blast ear / hearing damage
- Chemical/gas exposure
- Hypothermia
- Psychological first aid

**Priority 3 — Triage & communication:**
- How to triage multiple casualties
- What info to give when help arrives
- Building safety assessment
- Common mistakes that make injuries worse

**Format**: Step-by-step for non-medical people. "If X, do Y." Decision trees. Can be voice memos, docs, texts — any format.

---

## 10. TIMELINE & WORKSTREAMS

### Total: 45 days (April 3 → May 18, 2026)

---

### PHASE 1: FOUNDATION (Days 1-7) — April 3-9

**WS1: Immediate Setup (TODAY)**
- [x] Order all hardware — DONE ($311.24)
- [ ] Accept Gemma 4 license on HuggingFace (google/gemma-4-E4B-it)
- [ ] Join hackathon on Kaggle
- [ ] Create GitHub repo: `holos-medyk`
- [ ] Download Gemma 4 E4B-it weights on workstation
- [ ] Test baseline inference on workstation (text in → text out)
- [ ] Test native audio ASR on workstation with Ukrainian speech
- [ ] Install and test Piper TTS with Ukrainian voice on workstation
- [ ] **Call brother** — explain project, ask for "top 20 things"
- [ ] **Talk to girlfriend** — get her involved, start Ukrainian glossary

**WS2: Pi Setup (Sunday Apr 6)**
- [ ] Boot Pi from preloaded SD card
- [ ] Install LiteRT-LM on Pi
- [ ] Deploy Gemma 4 E4B-it quantized weights
- [ ] Verify inference: text → text
- [ ] Benchmark: tok/s, latency, memory
- [ ] Test USB mic + 3.5mm speaker
- [ ] Install Piper TTS on Pi

**WS3: End-to-End Voice Loop — MILESTONE 1 (by Apr 9)**
- [ ] Wire up: Mic → Gemma 4 ASR → Gemma 4 reasoning → Piper TTS → Speaker
- [ ] Test with Ukrainian prompts
- [ ] **SUCCESS CRITERIA**: Say something in Ukrainian → get spoken response

---

### PHASE 2: KNOWLEDGE & FINE-TUNING (Days 8-21) — April 10-23

**WS4: Emergency Knowledge Base**
- [ ] Brother's trauma protocols (English)
- [ ] ICRC/WHO guidelines
- [ ] TCCC adapted for civilians
- [ ] Ukrainian civil defense docs (girlfriend)
- [ ] Ukrainian medical glossary (girlfriend)
- [ ] Format as instruction/response pairs + multi-turn dialogues
- [ ] Structure as RAG-ready text chunks

**WS5: Fine-Tuning with Unsloth**
- [ ] Set up Unsloth (Colab Pro or local GPU)
- [ ] Prepare dataset (target: 500-2000 examples)
- [ ] Fine-tune Gemma 4 E4B with LoRA/QLoRA
- [ ] Quantize for LiteRT deployment
- [ ] Deploy to Pi, benchmark vs base model
- [ ] Brother tests medical accuracy
- [ ] Girlfriend tests Ukrainian naturalness
- [ ] Iterate on failures

**WS6: RAG / Grounding**
- [ ] Lightweight local RAG (SQLite + function calling)
- [ ] Pre-load shelter locations per city
- [ ] Test retrieval quality

---

### PHASE 3: SAFETY & ANDROID PORT (Days 22-30) — April 24 - May 2

**WS7: Medical Safety Guardrails**
- [ ] Safety rules: disclaimers, no diagnosis, no dosing
- [ ] Red-team the model (Kevin's specialty)
- [ ] Brother reviews red-team results
- [ ] Confidence scoring system

**WS8: Conversation Management**
- [ ] Triage flow: safety check → injury assessment → step-by-step guidance
- [ ] Multi-turn context
- [ ] "Repeat" and "speak slower" commands
- [ ] Calm, reassuring tone

**WS9: Android App**
- [ ] Port voice pipeline to Android using LiteRT-LM + ML Kit GenAI Prompt API
- [ ] Bundle fine-tuned E4B model weights in app (~4GB)
- [ ] Native Android TTS or bundled Piper for Ukrainian
- [ ] Simple UI: one big button to activate, transcript display
- [ ] Test on real Android device
- [ ] Also build E2B "lite mode" for older phones

---

### PHASE 4: DEMO & POLISH (Days 31-38) — May 3-10

**WS10: Web Dashboard (demo/video purposes)**
- [ ] React companion showing transcript, RAG retrieval, confidence scores
- [ ] Runs on laptop connected to Pi

**WS11: Physical Build & Testing**
- [ ] Pi: clean assembly, speaker, mic, battery bank
- [ ] Test full offline on both Pi and phone (airplane mode)
- [ ] Run 10+ emergency scenarios end-to-end on both devices
- [ ] Brother runs trauma scenarios
- [ ] Girlfriend runs Ukrainian scenarios
- [ ] Time each: activation → first spoken word

---

### PHASE 5: SUBMISSION (Days 39-46) — May 11-18

**WS12: Video Production (3 minutes)**
- [ ] Script:
  - **HOOK (0:00-0:20)**: Air raid siren. Dark. "Every day in Ukraine..."
  - **PROBLEM (0:20-0:50)**: Strike footage, stats. "Power goes out. Cell towers fail. People are injured."
  - **SOLUTION (0:50-1:40)**: Show the app on a phone — airplane mode, someone speaks Ukrainian, Gemma 4 responds with medical guidance. Then show the Pi device — same thing, for shelters. "Download it free. Or build one for a shelter for $75."
  - **TECHNICAL (1:40-2:10)**: Architecture. Fine-tuned with real trauma surgeon protocols. LiteRT on-device. Safety framework.
  - **IMPACT (2:10-2:50)**: "20 million Ukrainians have a smartphone. This app is free." Girlfriend or family testing. Brother's medical endorsement.
  - **CLOSE (2:50-3:00)**: "Голос Медик. A voice when no one else can answer."
- [ ] Upload to YouTube

**WS13: Written Submission**
- [ ] Kaggle Writeup (≤1,500 words)
- [ ] Public GitHub repo (`holos-medyk`)
- [ ] Model weights on HuggingFace
- [ ] Live demo: APK download or web simulator
- [ ] Cover image + media gallery

---

## 11. RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| E4B too heavy for older Ukrainian phones | Medium | Medium | Ship E2B "lite mode" as fallback |
| Gemma 4 E4B Ukrainian ASR quality poor | Medium | High | Fallback: Whisper-tiny for ASR |
| Fine-tuned model hallucinates medical advice | Medium | Critical | Brother reviews, guardrails, RAG grounding |
| English fine-tuning doesn't transfer to Ukrainian | Low | High | Add Ukrainian examples; test early |
| Android port takes longer than expected | Medium | High | Pi demo is the backup submission |
| Piper TTS Ukrainian voice sounds robotic | Medium | Medium | Test multiple voices, consider alternatives |
| LiteRT-LM + fine-tuned model compatibility | Medium | High | Test early, fallback to llama.cpp |
| Brother too busy for detailed protocols | Low | High | Use public TCCC/ICRC, have him review only |

---

## 12. CRITICAL PATH

1. **Gemma 4 E4B runs on Pi 5 via LiteRT-LM** — by Apr 6
2. **Ukrainian speech → text works** — by Apr 7
3. **Text → Ukrainian speech works** (Piper TTS) — by Apr 7
4. **End-to-end voice loop works** — by Apr 9
5. **Brother provides core protocols** — by Apr 15
6. **Fine-tuned model beats base model** — by Apr 23
7. **Android app works offline** — by May 2
8. **Video is compelling** — by May 18

If 1-4 work by end of Week 1, project is a go. Pi demo is the safety net if Android port (step 7) runs late.

---

## 13. IMMEDIATE NEXT STEPS (TONIGHT)

1. [x] ~~Order hardware~~ — DONE ($311.24)
2. [ ] Accept Gemma 4 license → https://huggingface.co/google/gemma-4-E4B-it
3. [ ] Join hackathon → https://www.kaggle.com/competitions/gemma-4-good-hackathon
4. [ ] Create GitHub repo: `holos-medyk`
5. [ ] Download Gemma 4 E4B-it weights on workstation
6. [ ] Test inference on workstation
7. [ ] **Call brother** — "20 things to save lives after a missile strike"
8. [ ] **Talk to girlfriend** — Ukrainian medical glossary, speech samples
9. [ ] Read LiteRT-LM docs → https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/

---

## 14. WHY THIS WINS

1. **Timing**: Gemma 4 launched yesterday. First to build E4B medical app on-device.
2. **Authenticity**: Girlfriend is Ukrainian, family lives this. Brother is a trauma surgeon. Real people, real stakes.
3. **Medical credibility**: Fine-tuned on protocols from a practicing trauma surgeon, not scraped web data. Brother validates outputs.
4. **Two-tier deployment**: Free app for 20M smartphones + $75 device for shelters. Complete coverage.
5. **Technical depth**: Fine-tuning (Unsloth) + edge deployment (LiteRT) + Android app (Cactus) + safety (red-teaming) + multimodal (voice I/O). Every Gemma 4 capability used.
6. **Track record**: Winner of OpenAI red-teaming challenge. MIT x3. IEEE published x2.
7. **Six tracks**: Main + Resilience + Cactus + LiteRT + Unsloth + Safety & Trust.
8. **Nothing like it exists**: Confirmed through exhaustive research.
9. **Beyond the hackathon**: Real product. Free app. Deployable now. NGOs and humanitarian orgs would use this.

---

*Голос Медик — A voice when no one else can answer.*
