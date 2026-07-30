"""Terminalgranssnitt for traningsspelet.

Flodet ar medvetet: du ser bara publik information nar du ska besluta — samma
information du har vid ett riktigt bord. Equity och EV visas forst *efter* att
du valt. Annars tranar du pa att lasa av en siffra istallet for att lasa spelet.
"""

import os
import random
import sys
from typing import Optional, Tuple

from trainer.bots import Bot, STYLES, make_table
from trainer.cards import RED_SUITS, pretty, suit_of
from trainer.coach import Coach, Feedback
from trainer.session import OpponentTracker, SessionStats, save_session
from trainer.table import Hand, Options, Player

# ---------- terminalfarger ----------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREY = "\033[90m"
    WHITE = "\033[97m"


def _enable_ansi() -> bool:
    """Sla pa ANSI-farger i Windows-konsolen. Returnerar False om det inte gick."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


class Palette:
    """Farglaggning som tyst faller tillbaka till ren text."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + C.RESET


# ---------- huvudklass ----------

class Trainer:
    def __init__(
        self,
        num_players: int = 6,
        starting_stack_bb: float = 100.0,
        big_blind: float = 100.0,
        seed: Optional[int] = None,
        color: bool = True,
        sims: int = 6000,
        auto_reset_stacks: bool = True,
    ):
        self.rng = random.Random(seed)
        self.bb = big_blind
        self.sb = big_blind / 2.0
        self.starting_stack = starting_stack_bb * big_blind
        self.auto_reset_stacks = auto_reset_stacks

        self.c = Palette(color and _enable_ansi())
        self.coach = Coach(sims=sims)
        self.stats = SessionStats()
        self.tracker = OpponentTracker()

        self.players, self.bots = make_table(
            "Du", self.starting_stack, num_players, self.rng
        )
        self.hero = self.players[0]
        self.button = self.rng.randrange(num_players)
        # Resultatet ackumuleras som netto per hand, sa baslinjen ar noll
        self.stats.chips_start = 0.0
        self.stats.chips_now = 0.0
        self.quit_requested = False

    # ---------- utskrift ----------

    def card_str(self, card: str) -> str:
        color = C.RED if suit_of(card) in RED_SUITS else C.CYAN
        return self.c(pretty(card), color, C.BOLD)

    def cards_str(self, cards) -> str:
        return " ".join(self.card_str(c) for c in cards) if cards else self.c("—", C.GREY)

    def rule(self, char: str = "─", width: int = 62) -> None:
        print(self.c(char * width, C.GREY))

    def show_table(self, hand: Hand) -> None:
        print()
        self.rule("═")
        header = (
            f" Hand #{self.stats.hands + 1}   "
            f"Blinds {self.sb:.0f}/{self.bb:.0f}   "
            f"{hand.street.upper()}   "
            f"Pott: {hand.pot:.0f} ({hand.pot / self.bb:.1f}bb)"
        )
        print(self.c(header, C.BOLD))
        print(f" Bord:  {self.cards_str(hand.board)}")
        self.rule()

        last_action = {}
        for a in hand.actions:
            if a.street == hand.street:
                if a.kind in ("bet", "raise", "all_in"):
                    last_action[a.player] = f"{a.kind} till {a.to_amount:.0f}"
                elif a.kind == "call":
                    last_action[a.player] = f"syn {a.amount:.0f}"
                else:
                    last_action[a.player] = a.kind

        for p in hand.players:
            marker = ">" if p is hand.next_to_act else " "
            tag = " (D)" if p.position == "BTN" else ""
            name = f"{p.name}{tag}"
            status = last_action.get(p.name, "")
            if p.folded:
                line = f" {marker} {name:<14} {p.position:<4} {p.stack:>8.0f}   {self.c('foldade', C.GREY)}"
            elif p.all_in:
                line = f" {marker} {name:<14} {p.position:<4} {p.stack:>8.0f}   {self.c('ALL-IN', C.YELLOW, C.BOLD)}"
            else:
                line = f" {marker} {name:<14} {p.position:<4} {p.stack:>8.0f}   {status}"
            if p.is_hero:
                line = self.c(line, C.BOLD)
            print(line)

        self.rule()
        print(f" Dina kort:  {self.cards_str(self.hero.hole)}   "
              f"{self.c('(' + self.hero.position + ')', C.GREY)}")
        self.rule("═")

    # ---------- inmatning ----------

    def prompt_action(self, hand: Hand, opts: Options) -> Optional[Tuple[str, float]]:
        """Las hjaltens handling. None = avsluta sessionen."""
        choices = []
        if opts.call_amount > 0:
            choices.append(self.c("[f]", C.BOLD) + "old")
        if opts.can_check:
            choices.append(self.c("[k]", C.BOLD) + "olla (check)")
        if opts.call_amount > 0:
            choices.append(self.c("[c]", C.BOLD) + f" syna {opts.call_amount:.0f}")
        if opts.can_raise:
            verb = "höj" if opts.call_amount > 0 else "beta"
            choices.append(self.c("[r]", C.BOLD) + f" {verb}")
            choices.append(self.c("[a]", C.BOLD) + "ll-in")
        choices.append(self.c("[?]", C.BOLD) + " läsningar")
        choices.append(self.c("[q]", C.BOLD) + "uit")

        while True:
            print()
            if opts.call_amount > 0:
                pot_odds = opts.call_amount / (opts.pot + opts.call_amount)
                print(self.c(
                    f" Att syna kostar {opts.call_amount:.0f} i en pott på "
                    f"{opts.pot:.0f} — du behöver vinna {pot_odds:.0%}",
                    C.GREY,
                ))
            print("  " + "   ".join(choices))
            try:
                raw = input(self.c(" > ", C.BOLD, C.WHITE)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return None

            if not raw:
                continue
            cmd, _, arg = raw.partition(" ")
            arg = arg.strip()

            if cmd in ("q", "quit", "avsluta"):
                return None
            if cmd in ("?", "info", "reads"):
                self.show_reads()
                continue
            if cmd in ("f", "fold") and opts.call_amount > 0:
                return ("fold", 0.0)
            if cmd in ("k", "check") and opts.can_check:
                return ("check", 0.0)
            if cmd in ("c", "call", "syn") and opts.call_amount > 0:
                return ("call", 0.0)
            if cmd in ("a", "allin", "all-in") and opts.can_raise:
                return ("all_in", 0.0)
            if cmd in ("r", "raise", "bet", "b") and opts.can_raise:
                amount = self.prompt_raise_amount(hand, opts, arg)
                if amount is not None:
                    return ("raise", amount)
                continue

            print(self.c("  Ogiltigt val.", C.YELLOW))

    def prompt_raise_amount(self, hand: Hand, opts: Options, arg: str = "") -> Optional[float]:
        """Tolka storlek. Accepterar tal, '1/2', '2/3', 'pot', '3bb'."""
        pot = opts.pot
        presets = {
            "1/2": pot * 0.5, "half": pot * 0.5, "h": pot * 0.5,
            "2/3": pot * 0.66, "3/4": pot * 0.75,
            "pot": pot, "p": pot,
        }
        while True:
            if not arg:
                print(self.c(
                    f"  Storlek {opts.min_raise_to:.0f}–{opts.max_raise_to:.0f}"
                    f"   (eller 1/2, 2/3, 3/4, pot, 3bb — tomt = avbryt)", C.GREY))
                try:
                    arg = input(self.c("  storlek > ", C.BOLD)).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return None
                if not arg:
                    return None

            target = None
            if arg in presets:
                target = opts.current_bet + presets[arg]
            elif arg.endswith("bb"):
                try:
                    target = float(arg[:-2].replace(",", ".")) * self.bb
                except ValueError:
                    target = None
            else:
                try:
                    target = float(arg.replace(",", "."))
                except ValueError:
                    target = None

            if target is None:
                print(self.c("  Förstod inte storleken.", C.YELLOW))
                arg = ""
                continue

            target = min(max(target, opts.min_raise_to), opts.max_raise_to)
            return target

    def show_reads(self) -> None:
        print()
        print(self.c(" Läsningar på bordet:", C.BOLD))
        reads = self.tracker.reads()
        if not reads:
            print(self.c("   Inga händer spelade än.", C.GREY))
        for line in reads:
            print(f"   {line}")

    # ---------- feedback ----------

    def show_feedback(self, fb: Feedback) -> None:
        color = {"ratt": C.GREEN, "narapa": C.YELLOW, "misstag": C.RED}[fb.verdict]
        symbol = {"ratt": "✓", "narapa": "~", "misstag": "✗"}[fb.verdict]

        print()
        print(self.c(f"  {symbol} {fb.headline}", color, C.BOLD))
        if fb.verdict != "ratt":
            print(f"    Du valde:    {fb.chosen_label}")
            print(f"    Bättre:      {self.c(fb.best_label, color, C.BOLD)}")
            if fb.ev_loss_bb > 0:
                print(f"    Kostnad:     {self.c(f'{fb.ev_loss_bb:.2f}bb', color)}")
        for line in fb.lines:
            print(self.c(f"    {line}", C.GREY))
        if fb.explanation:
            print(self.c(f"    Motorn: {fb.explanation}", C.GREY))
        if fb.concept:
            print()
            print(self.c(f"    → {fb.concept}", C.CYAN))

    # ---------- spelflode ----------

    def play_hand(self) -> bool:
        """Spela en hand. Returnerar False om anvandaren vill sluta."""
        if self.auto_reset_stacks:
            for p in self.players:
                p.stack = self.starting_stack

        hand = Hand(self.players, self.button, self.sb, self.bb, self.rng)
        shown_street = None

        while not hand.finished:
            player = hand.next_to_act
            opts = hand.options()

            if player.is_hero:
                if hand.street != shown_street:
                    shown_street = hand.street
                self.show_table(hand)

                choice = self.prompt_action(hand, opts)
                if choice is None:
                    self.quit_requested = True
                    return False

                kind, amount = choice
                villain = self.tracker.main_villain(hand, self.hero.name)
                fb = self.coach.review(hand, self.hero, opts, kind, amount, villain)
                self.stats.record(fb)
                self.show_feedback(fb)

                try:
                    hand.act(kind, amount)
                except ValueError as exc:
                    print(self.c(f"  {exc}", C.YELLOW))
                    continue
            else:
                bot = self.bots[player.name]
                kind, amount = bot.decide(hand, player, opts)
                hand.act(kind, amount)

        self.finish_hand(hand)
        return True

    def finish_hand(self, hand: Hand) -> None:
        self.stats.hands += 1
        self.tracker.observe_hand(hand, self.hero.name)

        print()
        self.rule("═")
        if hand.board:
            print(f" Bord:  {self.cards_str(hand.board)}")
        for line in hand.showdown_summary():
            print(self.c(f"   {line}", C.GREY))
        for res in hand.results:
            winners = " & ".join(res.winners)
            print(f" {self.c(winners, C.BOLD)} vinner {res.amount:.0f} — {res.description}")

        net = self.hero.won - self.hero.total_bet
        color = C.GREEN if net > 0 else (C.RED if net < 0 else C.GREY)
        print(f" Ditt resultat: {self.c(f'{net:+.0f} chips ({net / self.bb:+.1f}bb)', color, C.BOLD)}")
        self.stats.chips_now += net
        self.rule("═")

        self.button = (self.button + 1) % len(self.players)

    def show_summary(self) -> None:
        print()
        self.rule("═")
        print(self.c(" SESSIONSRAPPORT", C.BOLD))
        self.rule()
        for line in self.stats.summary_lines(self.bb):
            print(f" {line}")
        reads = self.tracker.reads()
        if reads:
            print()
            print(" Motståndarna du mötte:")
            for line in reads:
                print(f"   {line}")
        self.rule("═")

    def run(self, max_hands: Optional[int] = None) -> None:
        print()
        print(self.c("  POKERTRÄNING", C.BOLD, C.CYAN))
        print(self.c(
            "  Du får omedelbar feedback på varje beslut. Equity och EV visas\n"
            "  först efter att du valt — precis som vid ett riktigt bord.",
            C.GREY,
        ))

        played = 0
        while not self.quit_requested:
            if max_hands is not None and played >= max_hands:
                break
            if not self.play_hand():
                break
            played += 1

        self.show_summary()
