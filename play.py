"""Pokertraning — spela mot botar och fa direkt feedback pa varje beslut.

    python play.py                    # 6-max, 100bb stackar
    python play.py --players 3        # kortare bord
    python play.py --hands 20         # avsluta efter 20 hander
    python play.py --seed 42          # samma kort varje gang (bra for ovning)
    python play.py --keep-stacks      # lat stackarna folja med mellan hander
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.cli import Trainer
from trainer.session import save_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pokertraning med direkt feedback pa varje beslut."
    )
    parser.add_argument("--players", type=int, default=6, choices=range(2, 7),
                        help="Antal spelare vid bordet (2-6, standard 6)")
    parser.add_argument("--stack", type=float, default=100.0,
                        help="Starstack i big blinds (standard 100)")
    parser.add_argument("--bb", type=float, default=100.0,
                        help="Big blind i chips (standard 100)")
    parser.add_argument("--hands", type=int, default=None,
                        help="Avsluta automatiskt efter N hander")
    parser.add_argument("--seed", type=int, default=None,
                        help="Slumpfro — samma varde ger samma hander igen")
    parser.add_argument("--sims", type=int, default=6000,
                        help="Monte Carlo-simuleringar per equity-berakning")
    parser.add_argument("--keep-stacks", action="store_true",
                        help="Nollstall inte stackarna mellan hander")
    parser.add_argument("--no-color", action="store_true",
                        help="Stang av fargkodning")
    parser.add_argument("--save", metavar="FIL", default="trainer_sessions.json",
                        help="Fil att spara sessionsstatistik i")
    args = parser.parse_args()

    # Windows-konsolen behover UTF-8 for kortsymbolerna
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            os.system("chcp 65001 > nul")

    trainer = Trainer(
        num_players=args.players,
        starting_stack_bb=args.stack,
        big_blind=args.bb,
        seed=args.seed,
        color=not args.no_color,
        sims=args.sims,
        auto_reset_stacks=not args.keep_stacks,
    )

    try:
        trainer.run(max_hands=args.hands)
    except KeyboardInterrupt:
        print()
        trainer.show_summary()

    if args.save and trainer.stats.decisions:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save)
        try:
            save_session(path, trainer.stats, trainer.tracker)
            print(f"\n Sparat i {args.save}")
        except OSError as exc:
            print(f"\n Kunde inte spara sessionen: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
