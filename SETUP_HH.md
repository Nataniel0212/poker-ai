# Aktivera Hand History i PokerStars.SE

## Status
- HH-parsern är byggd och testad — fungerar perfekt
- PokerStars spelar händer (hand #259798948658 etc syns i loggen)
- Men HH-filer skrivs INTE till disk — inställningen saknas

## Vad som behövs
Hitta och aktivera "Save My Hand History" i PokerStars.SE-klienten.

## Möjliga platser i menyn
PokerStars.SE använder Locale=13 (svenska). Prova:

1. **Kugghjulet (⚙️)** → "Inställningar" / "Alternativ" / "Options"
   - Leta efter: "Spelhistorik" / "Handhistorik" / "Playing History" / "Hand History"
   - Bocka i: "Spara mina händer" / "Save My Hand History"

2. **Högerklicka på bordet** → "Handhistorik" / "Hand History"

3. **Menyrad** (om den finns) → "Alternativ" → "Handhistorik"

4. **Settings → Global** — ibland finns det under en "Global" eller "General" flik

5. **Instant Hand History-fönstret** — om du öppnar det kan det finnas en "Save" knapp

## Mapp att välja
```
C:\Users\natan\AppData\Local\PokerStars.SE\HandHistory
```
(Mappen finns redan, skapad av oss)

## Verifiering
Efter aktivering + en spelad hand, kör:
```
python tests/test_hh_live.py
```
Eller kolla manuellt:
```
dir C:\Users\natan\AppData\Local\PokerStars.SE\HandHistory\
```
Det ska dyka upp filer som `HH20260220 Bordnamn.txt`

## Alternativ om det inte går
Om PokerStars.SE inte stödjer lokal HH-sparning:
- Kolla om det finns en **"Request Hand History"** via email (vanligt på .SE/.EU)
- Använd **PokerStars "Instant Hand History"**-fönstret och copy-paste
- Kolla PokerStars support: https://www.pokerstars.se/help/
- Sista utvägen: fortsätt med OCR-baserad läsning (befintlig vision-kod finns kvar)

## Teknisk info
- PokerStars-användare: `Nataniel02`
- user.ini har `[LocalHH]` sektion men ingen synlig sökväg/toggle
- Loggen visar `Storing history item for hand { 259798948658 }` — intern lagring funkar
- Filen `user.ini` ändrades INTE efter att du "aktiverade" — antingen sparades det inte, eller inställningen finns på annan plats
