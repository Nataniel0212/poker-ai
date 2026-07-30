# Pokerträning med direkt feedback

Ett träningsspel där du spelar No-Limit Hold'em mot botar och får omedelbar
bedömning av varje beslut — innan nästa kort delas ut.

## Kom igång

```bash
python play.py
```

Inga nya beroenden behövs. Träningen använder bara `phevaluator` och `treys`,
som redan är installerade. Den rör varken OpenCV, Tesseract, PyQt6 eller Ollama.

```bash
python play.py --hands 20        # avsluta efter 20 händer
python play.py --players 3       # kortare bord
python play.py --seed 42         # samma händer varje gång — bra för att öva om en spot
python play.py --keep-stacks     # låt stackarna följa med mellan händer
python play.py --stack 40        # korta stackar (40bb)
```

## Varför inte skärmläsning?

Den gamla pipelinen försökte gissa spelläget genom att OCR-läsa skärmen, och
misslyckades ofta: potten lästes som `$52142` i en hand och `$7247` i nästa.

Här är träningsspelet självt bordet. Varje kort, varje stack och varje insats är
exakt känt. Ingen OCR, ingen fördröjning, inga feltolkade potter — och feedbacken
kommer på millisekunden istället för efter 660 ms av bildanalys.

## Hur bedömningen fungerar

Två olika mått används medvetet för de två faserna:

**Preflop bedöms mot range-charts.** Att öppna KJo från UTG är ett fel även om
just den handen råkar vinna. Equity mot slumpmässiga händer säger ingenting om
det — positionsbaserade ranges är rätt mått.

**Postflop bedöms mot EV.** När korten ligger på bordet är frågan konkret: vad
tjänar mest chips i längden, givet din equity och de odds du får? Coachen visar
hela EV-tabellen så du ser vad varje alternativ är värt, inte bara vilket som
vann.

Equity räknas med Monte Carlo (12 000 simuleringar per beslut, ~50 ms).

### Tre saker bedömningen medvetet gör

1. **Den visar inte equity förrän du valt.** Annars tränar du på att läsa av en
   siffra istället för att läsa spelet. Pot odds visas däremot i förväg — de kan
   du räkna ut vid ett riktigt bord.

2. **Den räknar med sin egen osäkerhet.** Equity är ett stickprov, inte ett
   facit. Ligger ditt val inom felmarginalen får det grönt ljus. En coach som
   säger "rätt" ena gången och "misstag" nästa på samma beslut går inte att lita
   på.

3. **Den föreslår inte marginella bluffar.** När en bluffhöjning och en fold är
   statistiskt oskiljbara rekommenderas folden. Modellens fold equity är en
   gissning, och att lära ut bluffar som bygger på dess svagaste antagande vore
   precis fel sak att träna in.

Fel *handling* (check istället för värdesatsning) räknas som ett misstag.
Fel *storlek* på rätt handling påpekas, men döms aldrig hårdare än "nästan".

## Motståndarna

Fem stilar, för att du ska lära dig känna igen dem:

| Stil | Beteende | Hur den slås |
|---|---|---|
| Nit | Bara premiumhänder | Stjäl blinds, respektera höjningar |
| TAG | Solid tight-aggressiv | Få uppenbara hål |
| LAG | Brett och aggressivt | Syna lättare |
| Station | Synar nästan allt | Bluffa aldrig, värdesatsa tunt |
| Maniac | Höjer med vad som helst | Låt den bluffa in i dina starka händer |

Tryck `?` under spel för att se profilerna som byggs upp medan du spelar (VPIP,
aggressionsfaktor och en läsning). Efter ungefär åtta händer mot samma spelare
börjar läsningen bli meningsfull.

## Sessionsrapporten

Direktfeedbacken lär dig rätt beslut i stunden. Rapporten svarar på den andra
frågan: *vilket* fel gör du om och om igen?

```
 Rätt:                3 (38%)
 EV förlorat:         20.7bb (10.33bb/hand)

 Träffsäkerhet per gata:
    preflop    3/3 (100%)
    flop       0/3 (0%)

 Dina största läckor:
    Postflop: missar värdesatsningar — 3 ggr, 12.4bb
    Postflop: betalar utan odds — 1 ggr, 3.5bb
```

Läckorna rankas efter vad de kostar, inte hur ofta de sker — tre dyra misstag är
viktigare att fixa än tio billiga. Varje session sparas i `trainer_sessions.json`
så du kan följa utvecklingen.

## Kommandon under spel

| Tangent | Gör |
|---|---|
| `f` | Fold |
| `k` | Check |
| `c` | Syna |
| `r` | Höj — sedan storlek: `600`, `1/2`, `2/3`, `3/4`, `pot`, `3bb` |
| `a` | All-in |
| `?` | Visa läsningar på motståndarna |
| `q` | Avsluta och visa rapporten |

Du kan skriva storleken direkt: `r pot`, `r 2/3`, `r 850`.

## Kod

| Fil | Ansvar |
|---|---|
| `trainer/table.py` | Spelmotorn — budgivning, sidopotter, showdown |
| `trainer/cards.py` | Kortlek och Monte Carlo-equity |
| `trainer/bots.py` | Motståndarnas beslutslogik |
| `trainer/coach.py` | Bedömningen och EV-modellen |
| `trainer/session.py` | Statistik, motståndarprofiler, läckrapport |
| `trainer/cli.py` | Terminalgränssnittet |

Återanvänder `strategy/engine.py` (range-charts) och `strategy/push_fold.py`
från det befintliga projektet.

## Tester

```bash
python -m unittest discover -s tests -p "test_trainer*.py"
```

22 tester. Motorn testas på invarianter som måste hålla oavsett hur botarna
spelar: chips får inte uppstå ur tomma intet, handen måste alltid ta slut, och
potten måste alltid delas ut i sin helhet — verifierat över 400 slumpade händer
plus 200 all-in-situationer med sidopotter.

Coachen testas mot spottar med entydigt facit (folda nutsen, syna med luft), och
mot att samma spot får samma betyg varje gång.

## Vad EV-siffrorna inte är

Uppskattningar, inte solverutdata. De antar rimlig fold equity och att handen
spelas ut rakt. De duger utmärkt för att skilja ett bra beslut från ett dåligt,
men lita inte på sista decimalen — och en enskild hand säger ingenting alls.
Det är kurvan över hundratals händer som betyder något.

Preflop-kostnaderna är grövre än postflop. Preflop bedöms mot charts, inte mot
EV, så det finns ingen exakt siffra att hämta — kostnaden uppskattas till ungefär
halva insatsen när du betalar in i en pott du borde ha lämnat. Siffran finns där
för att rangordna läckor mot varandra, inte för att vara exakt.
