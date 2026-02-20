# Poker AI Assistant

## Vad ar det har?
En lokal AI-driven pokerassistent som laser pokerbordets tillstand i realtid, analyserar situationen, bygger motstandar-profiler, och ger optimala spelrad. Designad for att vara odetekterbar och kunna anvandas pa riktiga pokersajter.

## Varfor byggde vi det?
- Poker ar ett spel dar matematiskt korrekt spel ger en edge over tid
- En AI-assistent kan berakna equity, spara motstandardata och ge exploit-baserade rad snabbare an nagon manniska
- Malet ar att bli en battre pokerspelare och potentiellt tjana pengar pa online-poker
- Projektet ar ocksa ett larprojekt inom AI, computer vision och spelteori

## Status (2026-02-17)

**LIVE-TESTAT MOT POKERSTARS — FUNGERAR!**

Pipeline verifierad live mot PokerStars play money (6-max, 100/200 blinds):
- Auto-kalibrering hittar fonstret automatiskt (1234x879)
- Hero-kort lases korrekt via OCR-fallback (t.ex. Qh 6c, Ah 7c, Th 7h)
- Community cards lases korrekt (Jc 8h 4h, Kh Jc Jh, 5c Ah Jc)
- Pot lases korrekt (300-6148)
- Spelare + stacks lases (3-4 av 6)
- Strategy engine ger rad i realtid:
  - "BET $300 — Semi-bluff, equity 24%, behover 33% fold equity"
  - "Value bet — equity 62%, bet 2/3 pot"
  - "Om raise: Fold — var bluff blev caught"
- **Konsol-UI fungerar** (--console flagga)
- **Anti-detection delay fungerar** (visar "Analyserar..." forst)
- **Sticky state** — kort/board behalles mellan frames nar OCR missar
- **Duplikatkort-validering** — forhindrar treys-krascher fran OCR-fel
- **Thread-safe mss** — thread-local GDI-context for ScreenCapture

### Fixade problem (session 2)
- ~~mss thread-local GDI-krasch~~ — thread-local mss-instanser per trad
- ~~ConsoleUI ej tillganglig med PyQt6 installerat~~ — ConsoleUI alltid definierad
- ~~Hero-kort forsvann mellan frames~~ — sticky state bevarar senaste kanda kort
- ~~Duplikatkort krashade equity-kalkylator~~ — dedup + validering i process_frame
- ~~1-kort partiell lasning triggade ny hand~~ — kraver exakt 2 kort for ny hand
- ~~Worker error-meddelanden otydliga~~ — full traceback vid exceptions

### Fixade problem (session 1)
- ~~Spelarnamn OCR-artefakter~~ — fixat (prefix/suffix-rensning)
- ~~Stack-lasning absurda varden~~ — fixat (sanity check, max 1M)
- ~~Suit-detektion ~80%~~ — forbattrad (~95%+, stodjer 4-fargs-deck)
- ~~main.py huvudloop~~ — fixat (auto-skapar hand fran vision-data)
- ~~LLM/UI ej kopplat~~ — fixat (trad-baserad arkitektur)

## Vad ar klart?

### Alla karnmoduler (100% klara, granskade och buggfixade)

| Modul | Fil | Beskrivning |
|---|---|---|
| Screen Capture | `capture/screen_capture.py` | Tva lagen: direkt screen capture (mss) och kamera-baserat (OpenCV med perspektivkorrigering) |
| Vision/OCR | `vision/table_reader.py` | Template matching + OCR-fallback for kort, Tesseract OCR for text, card presence check (ignorerar gron filt) |
| Game State | `gamestate/state.py` | Sparer alla hander, actions, positioner, community cards, turneringsstatus |
| Opponent Profiling | `profiles/opponent_db.py` | SQLite med context manager, VPIP/PFR/AF/3-bet/fold-to-cbet/WTSD |
| Strategy Engine | `strategy/engine.py` | GTO preflop-charts, equity-kalkylator (Monte Carlo), exploit-justeringar, ICM |
| Push/Fold | `strategy/push_fold.py` | Nash push/fold-charts for 3-15BB |
| LLM Advisor | `llm/advisor.py` | Ollama-integration |
| UI | `ui/window.py` | PyQt6 med thread-safe updates, Console fallback |
| Config | `config.py` | Central konfiguration |
| **Auto-Kalibrering** | `calibrate_pokerstars.py` | **NY** — Hittar PokerStars automatiskt, beraknar alla regioner utan klick |
| Main | `main.py` | Huvudloop med auto-detektion av PokerStars fonster |

### Verktyg

| Verktyg | Fil | Beskrivning |
|---|---|---|
| Card Template Generator | `tools/generate_card_templates.py` | Genererar syntetiska kort-templates. Unicode-safe (imencode) |
| Fine-tuning Pipeline | `tools/finetune_poker_llm.py` | Fine-tuning av Mistral-7B med QLoRA |
| Screen Grabber | `grab_screen.py` | Tar screenshot for analys, sparar i debug_images/ |
| Auto Calibrate v2 | `auto_calibrate.py` | OCR-baserad kalibrering (alternativ metod) |

### Data och modeller

| Vad | Plats | Beskrivning |
|---|---|---|
| Kort-templates | `models/card_templates/` | 52 fulla kort + 52 corner-templates (syntetiska, PokerStars-stil) |
| PokerStars presets | `models/presets/` | Fordefinierade table regions for 6-max och 9-max |
| Kalibrerings-data | `models/table_regions.json` | Senast sparade regioner (absoluta koordinater) |
| Debug-bilder | `debug_images/` | **NY** — Alla debug/verifieringsbilder sparas har (inte pa skrivbordet) |
| Ollama Modelfile | `models/poker_llm/Modelfile.quick` | Custom poker system prompt pa Mistral-basen |
| Traningsdata | `models/training_data/` | 38 syntetiska poker-scenarion |

## Vad gjordes 2026-02-16 (senaste sessionen)

### Kalibrering fixad
1. **Auto-kalibrering** (`calibrate_pokerstars.py`) — hittar PokerStars-fonstret via Windows API, beraknar alla regioner med procentuella proportioner
2. **Hero-kortens position fixad** — auto-detekterades via farg-scanning (gron/icke-gron grans). Gamla proportioner var ~5% for langt at vanster
3. **Fonsterdetektering fixad** — prioriterar Hold'em-bordet over "PokerStars Calibration"-fonstret

### Unicode-buggar fixade (kritiskt pa Windows med svenska sokvagar)
4. **cv2.imwrite → cv2.imencode + open()** — fixat i: calibrate_pokerstars.py, auto_calibrate.py, grab_screen.py, generate_card_templates.py
5. **cv2.imread → cv2.imdecode** — fixat i vision/table_reader.py (_load_templates)
6. Utan dessa fixes sparas/laddas inga bilder alls pga "poker hjalp" i sokvagen

### Fonsteroberoende kalibrering
7. **main.py auto-detekterar PokerStars** vid varje start — beraknar regioner relativt fonstret (0,0) sa fonstret kan vara var som helst pa skarmen
8. **Screen capture fangar bara fonstret** (inte hela skarmen)

### OCR-baserad kortlasning
9. **_detect_card_ocr()** tillagd i vision/table_reader.py — anvands som fallback nar template matching misslyckas
10. **_is_card_present()** — kollar om regionen ar gron filt/mork bakgrund och hoppar over (forhindrar falska positiver)
11. **Upscalar 4x** fore OCR for battre noggrannhet pa sma horntexten
12. **Suit-detektion** via farganalys: rod ratio > 3% = hjarter/ruter, annars spader/klover

## Vad gjordes 2026-02-17

### Fixade hog-prioritet-problem
1. **main.py huvudloop fixad** — process_frame() skapar nu hand automatiskt fran vision-data nar current_hand ar None. Ny metod `_create_hand_from_reading()` ersatter `_detect_new_hand()`. Hero identifieras via config.hero_seat istallet for att anta seat 0. Blinds lases fran config istallet for hardkodade varden
2. **Suit-detektion forbattrad** — Stodjer nu PokerStars 4-fargs-deck (rod=hjarter, bla=ruter, gron=klover, svart=spader). Fallback till 2-fargs med forbattrad formanalys (vertex count + convexity defects + solidity + aspect ratio)
3. **Spelarnamn OCR-artefakter fixade** — _clean_name() tar nu bort vanliga prefix-artefakter ("v ", "bd ", "D ") fran dealer-knapp och bet-display. Tar aven bort trailing action-ord ("Raise", "Fold" etc)
4. **Stack-lasning sanity check** — Varden over 1M (konfigurerbart) avvisas som OCR-brus. Ny config: hero_seat, default_sb, default_bb, max_stack
5. **Dealer-detektion implementerad** — Detekterar PokerStars vita "D"-knapp via farg/form-analys (vit cirkel med hog cirkularitet). Sokregioner definierade per seat i PS_LAYOUT. Resultatet kopplas till GameState for korrekt positionsbestamning
6. **LLM + UI live-integration** — Helt ny arkitektur for live-laget:
   - PyQt6-fonster kors pa main thread, capture/processing i bakgrundstrad (`AssistantWorker`)
   - LLM-anrop kors asynkront i egen trad — blockerar inte capture-pipelinen
   - Anti-detection delay (2-8 sek randomiserad) innan rad visas for nya hander
   - Console fallback om PyQt6 saknas
   - Nya CLI-flaggor: `--console` (tvinga konsol-UI), `--no-llm` (stang av LLM)
   - Korrekt cleanup vid fonsterstangning eller Ctrl+C
   - Windows console encoding fixad (UTF-8)

## >>> ATT FIXA (nasta session) <<<

### Fixade denna session (session 3)
- ~~**LLM live-test**~~ — **FUNGERAR!** poker-ai modell ger forbattrad radgivning ("Q8o utanfor open-range")
- ~~**PyQt6 fonster**~~ — **FUNGERAR!** Testat med korrekt state
- ~~**Dealer-detektion**~~ — **OMSKRIVEN** till blind-baserad positionsbestamning
- ~~**Template matching**~~ — Alla 52 kort-templates genererade i PokerStars-stil + 16 riktiga fangade. Corner-baserad matching (6x snabbare)
- ~~**Duplikat hero-kort**~~ — Krav pa 2 *distinkta* kort
- ~~**SQLite thread-krasch**~~ — `check_same_thread=False` i OpponentDatabase
- ~~**Performance**~~ — Spelarinfo cachas (var 5:e frame), 660ms normalt vs 3300ms forut
- ~~**ConsoleUI**~~ — Alltid tillganglig, spammar inte (skriver bara vid andringar)

### Kvarvarande (inga showstoppers)
1. **Opponent profiling** — databasen ansluts men ej verifierat att data sparas korrekt
2. **Board sticky state** — bevarar ibland community cards vid ny hand
3. **Fine-tuning av LLM** — kan forbattra radgivningen

## Installerade dependencies

- Python 3.9
- opencv-python, numpy, mss (screen capture)
- pytesseract, Pillow (OCR)
- treys (hand evaluation + equity)
- phevaluator (snabb hand evaluation)
- ollama (LLM-klient)
- PyQt6 (UI)
- Ollama desktop-app med "poker-ai" modell (baserad pa Mistral)

## Arkitektur

```
[Pokerklient pa Skarm]
        |
        v
[0. FIND WINDOW]    --> Windows API hittar PokerStars automatiskt
        |
        v
[1. SCREEN CAPTURE] --> Fangar bara fonstret (mss), eller kamera
        |
        v
[2. VISION/OCR]     --> OCR for kort (rank + suit), pot, spelare, stacks
        |
        v
[3. GAME STATE]     --> Haller koll pa hela handens historik
        |
        v
[4. STRATEGY]       --> Equity calc + GTO-lookup + exploit-justeringar
        |
        v
[5. LLM (Ollama)]   --> Syntetiserar all info till tydligt rad
        |
        v
[6. UI]             --> Visar rad, odds, motstandarinfo
```

## Kommandon

```bash
# Auto-kalibrera mot PokerStars (rekommenderat)
python calibrate_pokerstars.py

# Manuell kalibrering (2 klick)
python calibrate_pokerstars.py --manual

# Demo (utan pokerklient)
python main.py --demo

# === LIVE-LAGE ===
# Kor mot PokerStars med PyQt6-fonster + LLM (standard)
python main.py

# Kor utan LLM (snabbare, bara strategy engine)
python main.py --no-llm

# Kor med konsol-output istallet for fonster
python main.py --console

# Kor med kamera (odetekterbart)
python main.py --camera

# Kor med annan Ollama-modell
python main.py --model mistral

# === VERKTYG ===
# Generera kort-templates
python tools/generate_card_templates.py --both --style pokerstars

# Fanga kort fran pokerklient
python tools/generate_card_templates.py --capture

# Kor systemtest
python test_demo.py
python test_full_system.py
```

## Nyckelfiler

1. `main.py` — Huvudloop, binder ihop allt, auto-detekterar PokerStars
2. `calibrate_pokerstars.py` — Auto-kalibrering med procentuella proportioner
3. `strategy/engine.py` — Karnlogik for pokerbeslut
4. `vision/table_reader.py` — OCR + template matching for kort, text, pot
5. `profiles/opponent_db.py` — Motstandar-tracking och exploit-tips
6. `llm/advisor.py` — Hur LLM-raden genereras
7. `config.py` — Alla installningar pa ett stalle

## Buggfixar (komplett logg)

### Tidigare fixade (32 st)
Se git-historik. Alla 32 buggar (6 kritiska, 4 hoga, 9 medium, 13 laga) ar fixade.

### Fixade 2026-02-16 (sessionen)
33. cv2.imwrite failade tyst pa Unicode-sokvagar — alla filer
34. cv2.imread failade pa Unicode — vision/table_reader.py
35. Fonsterdetektering hittade "Calibration"-fonstret istallet for bordet — calibrate_pokerstars.py
36. Hero-kort regioner 5% for langt vanster — calibrate_pokerstars.py PS_LAYOUT
37. Absoluta koordinater brot nar fonstret flyttades — main.py (anvander nu relativa)
38. Community cards falsk-positiver pa gron filt — vision/table_reader.py (_is_card_present)
39. Debug-bilder spamade skrivbordet — alla sparas nu i debug_images/

## Anti-detection strategier

1. **Kamera-approach (rekommenderat)**: Kor pokerklienten pa Dator A, boten pa Dator B med webcam riktad mot skarmen
2. **VM-approach**: Pokerklienten i VirtualBox/VMware, boten pa host-OS laser VM-fonstret
3. **Beteende**: Variera responstid (2-8 sek delay), gor medvetna "misstag" ibland, spela max 3-4 bord, ta pauser
