"""Pokerrad fran skarmen — fristaende program.

    python watch.py --calibrate          # forsta gangen pa en ny sajt
    python watch.py                      # kor radgivningen

Kraver inget Claude Code och ingenting sajtspecifikt i koden. Vad korten pa
just din sajt ser ut som ligger i en profil som kalibreringen skapar.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenadvisor import capture, overlay
from screenadvisor.calibrate import calibrate_from_images, calibrate_live
from screenadvisor.profile import Profile


def cmd_list() -> int:
    names = Profile.list_all()
    if not names:
        print(" Inga profiler an. Skapa en med:  python watch.py --calibrate")
        return 0
    print(" Profiler:")
    for name in names:
        profile = Profile.load(name)
        print(f"   {name:<22} {profile.status()}")
    return 0


def cmd_test_image(profile: Profile, path: str, opponents: int) -> int:
    """Kor lasningen mot en sparad bild — bra for att verifiera en profil."""
    import cv2
    import numpy as np

    from screenadvisor.advice import build_advice
    from screenadvisor.detect import annotate
    from screenadvisor.reader import read_table

    if not os.path.exists(path):
        print(f" Hittar inte {path}")
        return 1
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print(f" Kunde inte lasa {path}")
        return 1

    read = read_table(frame, profile.templates,
                      hero_zone=profile.hero_zone,
                      opponents_override=opponents)
    print(f" Dina kort:   {' '.join(read.hero) if read.hero else '—'}")
    print(f" Bord:        {' '.join(read.board) if read.board else '—'}")
    print(f" Motstandare: {read.opponents}")
    if read.unknown_cards:
        print(f" Okanda kort: {read.unknown_cards}")
    if read.note:
        print(f" Notering:    {read.note}")

    if read.usable:
        advice = build_advice(read.hero, read.board, read.opponents)
        print()
        print(f" >> {advice.headline}")
        for line in advice.lines:
            print(f"    {line}")

    out = os.path.splitext(path)[0] + "_lasning.png"
    ok, buf = cv2.imencode(".png", annotate(frame, read.candidates))
    if ok:
        buf.tofile(out)
        print(f"\n Markerad bild sparad: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ger pokerrad genom att lasa spelet pa din skarm.",
    )
    parser.add_argument("--profile", default="standard",
                        help="Sajtprofil att anvanda (standard: 'standard')")
    parser.add_argument("--calibrate", action="store_true",
                        help="Kor kalibreringen: valj region och lar in korten")
    parser.add_argument("--region", action="store_true",
                        help="Valj om skarmregionen, behall inlarda kort")
    parser.add_argument("--list", action="store_true",
                        help="Visa sparade profiler och deras status")
    parser.add_argument("--from-images", nargs="+", metavar="BILD",
                        help="Kalibrera fran sparade skarmdumpar")
    parser.add_argument("--test-image", metavar="BILD",
                        help="Testa profilen mot en sparad bild")
    parser.add_argument("--opponents", type=int, default=None,
                        help="Antal motstandare kvar i handen")
    parser.add_argument("--table-size", type=int, default=None,
                        help="Antal spelare vid bordet (sparas i profilen)")
    parser.add_argument("--console", action="store_true",
                        help="Textlage istallet for fonster")
    parser.add_argument("--sims", type=int, default=8000,
                        help="Monte Carlo-simuleringar per equity-berakning")
    parser.add_argument("--interval", type=float, default=0.4,
                        help="Sekunder mellan skarmlasningar")
    args = parser.parse_args()

    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    if args.list:
        return cmd_list()

    profile = Profile.load(args.profile)
    if args.table_size:
        profile.table_size = args.table_size
        profile.save()

    if args.from_images:
        calibrate_from_images(profile, args.from_images)
        return 0

    if args.calibrate or args.region:
        if args.region:
            profile.region = None
        calibrate_live(profile)
        return 0

    if args.test_image:
        return cmd_test_image(profile, args.test_image,
                              args.opponents or profile.default_opponents)

    # Kor pa riktigt
    if profile.region is None:
        print(" Den har profilen har ingen skarmregion an.")
        print(f" Kor forst:  python watch.py --calibrate --profile \"{profile.name}\"")
        return 1

    missing_ranks, missing_suits = profile.templates.missing()
    if missing_ranks or missing_suits:
        print(f" Obs: profilen ar inte fardigkalibrerad ({profile.status()}).")
        print(" Kort med glyfer som inte lards in an visas som okanda —")
        print(" programmet gissar aldrig. Kor --calibrate for att fylla pa.")
        print()

    if args.console:
        return overlay.run_console(profile, args.opponents,
                                   sims=args.sims, interval=args.interval)
    return overlay.run_overlay(profile, args.opponents,
                               sims=args.sims, interval=args.interval)


if __name__ == "__main__":
    sys.exit(main())
