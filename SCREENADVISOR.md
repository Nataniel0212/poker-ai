# Pokerråd från skärmen

Ett fristående program som läser pokerspelet på din skärm och visar råd i ett
fönster ovanpå. Byggt för gratis enspelarspel mot datorn, som 247 Free Poker.

Kräver inget Claude Code. Ingenting i koden är bundet till en specifik sajt.

## Snabbstart

```bash
python watch.py --calibrate --profile "247 free poker"
```

Kalibreringen gör två saker: du drar en ruta runt pokerbordet, och sedan får
programmet lära sig hur just den sajtens kort ser ut. Därefter:

```bash
python watch.py --profile "247 free poker"
```

Ett fönster dyker upp som ligger kvar ovanpå spelet och uppdateras medan du
spelar.

Profilen `247 free poker` finns redan med **2, 7, K, klöver och hjärter**
inlärda från dina egna skärmdumpar. Resten fyller du på under kalibreringen.

## Varför det här fungerar när OCR inte gjorde det

Det gamla försöket lät Tesseract gissa sig fram i renderad text. Resultatet syns
i dina egna loggar: potten lästes som `$52142` i en hand och `$7247` i nästa,
motståndarnamn blev `6948`.

Det här programmet gör tre saker annorlunda.

**Det letar efter hörnindexet, inte kortets kontur.** Två överlappande kort
smälter ihop till en enda vit klump som inte går att dela pålitligt — det var
sannolikt en av grundorsakerna till att kort missades. Men hörnindexet, alltså
rankbokstaven med färgsymbolen strax under, är alltid synligt. Det är hela
poängen med att fjädra korten. Ett hörnindex = ett kort.

**Det matchar mot inlärda glyfer istället för att tolka text.** Efter
kalibreringen jämförs varje glyf med en pixelkopia av samma glyf från din sajt.
På dina skärmdumpar ger det säkerheten **1.00** för riktiga kort, medan
chipsstaplar och pott-text hamnar på **0.31 eller lägre** mot en gräns på 0.62.
Det är ingen knapp marginal — det är två helt skilda världar.

**Det läser inte siffror.** Pott och stackar är precis det OCR är sämst på.
Istället visas vilken equity du behöver för att syna olika insatsstorlekar. Du
ser själv vad motståndaren satsat, och den jämförelsen kan aldrig bli fel på
grund av en feltolkad siffra.

## Grundregeln: gissa aldrig

Ett kort som inte kan läsas säkert rapporteras som okänt. Det står `2 kort kunde
inte läsas` i fönstret och du får inget råd.

Det är avsiktligt och det är den viktigaste egenskapen i hela programmet. Ett
tomrum i läsningen märker du direkt och kan åtgärda. Ett påhittat kort ger dig
ett självsäkert råd byggt på en hand du inte har — och det märker du först när
du redan agerat på det.

## Prestanda

| Steg | Tid |
|---|---|
| Skärmfångst | 5 ms |
| Kortläsning | 8 ms |
| Equity (8 000 simuleringar) | 170 ms |

Equity räknas bara om när korten faktiskt ändras, så mellan besluten kostar
läsningen 13 ms. Det gamla systemet låg på 660 ms per bildruta.

## Kalibrering i detalj

Du behöver inte mata in alla 52 korten — bara 13 rankar och 4 färger, alltså 17
glyfer. De dyker upp av sig själva medan du spelar några händer.

```bash
python watch.py --calibrate --profile "min sajt"
```

1. Dra en ruta runt bordet, tryck Enter. Ta med hela bordet men inte mer.
2. Spela en hand. Tryck Enter i terminalen när du har kort framme.
3. Programmet visar varje glyf det inte känner igen, stort och i sitt
   sammanhang. Skriv vad det är: `Kh`, `10s`, `7c`.
4. Är det inte ett kort — tryck bara Enter. Programmet föreslår ibland
   chipsstaplar och stora färgsymboler; de hoppar du över.
5. Upprepa till statusraden säger `Komplett`.

Går det snett kan du köra om regionen utan att tappa inlärda kort:

```bash
python watch.py --region --profile "min sajt"
```

Du kan också kalibrera från sparade skärmdumpar istället för direkt från
skärmen:

```bash
python watch.py --from-images bild1.png bild2.png --profile "min sajt"
```

## En sajt till

Samma kod, ny profil. Inget behöver ändras i källkoden.

```bash
python watch.py --calibrate --profile "annan sajt"
python watch.py --profile "annan sajt"
python watch.py --list
```

## Motståndare

Antalet motståndare är en **inställning**, inte en läsning. Kortryggar överlappar
olika mycket i olika spel, så en räknare blir antingen för hög eller för låg —
och equity beror märkbart på siffran. Hellre en siffra du styr än en som ser
automatisk ut och är fel.

Justera med **−** och **+** i fönstret när spelare foldar. Utgångsvärdet är
bordsstorleken minus dig:

```bash
python watch.py --table-size 5 --profile "247 free poker"
```

## Kommandon

| Flagga | Gör |
|---|---|
| `--calibrate` | Välj region och lär in korten |
| `--region` | Välj om regionen, behåll inlärda kort |
| `--from-images ...` | Kalibrera från sparade bilder |
| `--test-image BILD` | Testa profilen mot en bild, sparar en markerad kopia |
| `--list` | Visa profiler och deras status |
| `--opponents N` | Antal motståndare kvar |
| `--table-size N` | Antal spelare vid bordet |
| `--console` | Textläge istället för fönster |
| `--sims N` | Simuleringar per equity-beräkning (standard 8 000) |
| `--interval S` | Sekunder mellan skärmläsningar (standard 0.4) |

Att felsöka en profil mot en sparad bild är ofta snabbaste vägen:

```bash
python watch.py --test-image minbild.png --profile "247 free poker"
```

Det skriver ut vad som lästes och sparar `minbild_lasning.png` med grön ruta
runt varje läst kort och orange runt förkastade förslag.

## Kod

| Fil | Ansvar |
|---|---|
| `screenadvisor/detect.py` | Hittar hörnindex i en bildruta |
| `screenadvisor/glyphs.py` | Normalisering, mall-arkiv, klassificering |
| `screenadvisor/reader.py` | Sätter ihop hjältekort, bord och motståndare |
| `screenadvisor/advice.py` | Equity, drag, outs, råd |
| `screenadvisor/capture.py` | Skärmfångst och regionval |
| `screenadvisor/calibrate.py` | Inlärningen |
| `screenadvisor/overlay.py` | Fönstret som ligger ovanpå |
| `screenadvisor/profile.py` | Sparade sajtprofiler |
| `watch.py` | Startpunkt |

Återanvänder `strategy/engine.py` för preflop-ranges och `trainer/cards.py` för
equity.

## Tester

```bash
python -m unittest tests.test_screenadvisor
```

23 tester som körs mot dina riktiga skärmdumpar. De viktigaste:

- **Kort som inte lärts in gissas aldrig.** Med en profil som kan 7 och 2 men
  inte K läses KK-bilden inte som något — den rapporteras som okänd.
- **Glyfer generaliserar över storlek.** Mallar inlärda i 755 px bredd läser
  korrekt i 716 px, så läsningen faller inte ihop om fönstret ändrar storlek.
- **Brus räknas inte som olästa kort.** Annars skulle verktyget ständigt hävda
  att det inte kan läsa bordet trots att alla riktiga kort är lästa.

## Begränsningar

Programmet läser kort. Det läser inte pott, insatser, position eller vem som
agerat — och låtsas inte göra det.

Det betyder att råden är equity-baserade och att du själv bidrar med
sammanhanget: hur stor insatsen är, var du sitter, hur många som är kvar. Det är
en medveten avvägning. De delarna kräver siffer-OCR, och siffer-OCR är exakt vad
som sänkte förra bygget.

Preflop-råden säger vilka positioner handen är spelbar från istället för att
gissa var du sitter. Du vet redan var du sitter.
