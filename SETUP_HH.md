# Aktivera Hand History i PokerStars.SE

## Status
- HH-parsern är byggd och testad — fungerar perfekt
- PokerStars spelar händer (hand #259798948658 etc syns i loggen)
- Men HH-filer skrivs INTE till disk — inställningen saknas

## Officiell väg i menyn (bekräftad via PokerStars support)
**Inställningar → Spelhistorik → Handhistorik** (eng: Settings → Game History → Hand History)
Bocka i **"Spara min handhistorik"** och välj mapp.

Obs: menyn heter "Game History"/"Spelhistorik", inte "Playing History" som äldre
guider säger.

## Officiell felsökning om inställningen inte sparas
Källa: https://www.pokerstars.se/help/articles/win-hh-not-saving-initial/

1. **Verifiera** att "Spara min handhistorik" faktiskt är ibockad
2. **Byt lagringsmapp till Dokument** — skrivskyddade mappar (Program Files)
   blockerar tyst
3. **Trasig user.ini**: Inställningar → Hjälp → "Öppna min inställningsmapp",
   döp om `user.ini` till `user.old`, starta om klienten, aktivera igen.
   (Detta nollställer övriga klientinställningar — ta backup först.)
4. **Stäng tredjepartsprogram** (HM/PT etc) som kan störa

## Verifiera att inställningen fastnade
`[LocalHH]`-sektionen i user.ini innehåller en binär `state=`-blob.
Baseline (AV, 2026-07-30): `state=0100000000000000000000000002000000020000`
Om blobben ändrats efter att klienten stängts har inställningen sparats.

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
