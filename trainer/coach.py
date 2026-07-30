"""Coachen — bedomer ditt beslut i samma sekund du fattar det.

Tva olika verktyg anvands medvetet for de tva faserna:

* **Preflop** avgors mot range-charts. Att oppna KJo fran UTG ar ett fel aven
  om just den handen rakar vinna, och equity mot slumpmassiga hander sager
  ingenting om det. Charts ar ratt matt.
* **Postflop** avgors mot EV. Nar korten ar pa bordet ar fragan konkret: vad
  tjanar mest chips i langden, givet din equity och de odds du far?

EV-siffrorna ar uppskattningar — de antar rimlig fold equity och att handen
spelas ut rakt. De duger utmarkt for att skilja ett bra beslut fran ett daligt,
men lita inte pa sista decimalen.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from strategy.engine import StrategyEngine
from trainer.cards import equity as calc_equity, hand_notation
from trainer.table import Hand, Options, Player

# Hur stort EV-tapp som skiljer ett bra beslut fran ett daligt (i big blinds)
CLOSE_ENOUGH_BB = 0.15
MINOR_MISTAKE_BB = 0.80


@dataclass
class Feedback:
    verdict: str                  # "ratt" | "narapa" | "misstag"
    headline: str
    chosen_label: str
    best_label: str
    explanation: str
    equity: float = 0.0
    pot_odds: float = 0.0
    ev_loss_bb: float = 0.0
    lines: List[str] = field(default_factory=list)
    concept: str = ""             # vad spotten larde ut
    category: str = ""            # lackkategori, for sammanstallningen efterat
    street: str = "preflop"

    @property
    def is_correct(self) -> bool:
        return self.verdict == "ratt"


def _fmt(amount: float, bb: float) -> str:
    return f"{amount:.0f} ({amount / bb:.1f}bb)" if bb else f"{amount:.0f}"


class Coach:
    def __init__(self, sims: int = 12000):
        self.engine = StrategyEngine()
        self.sims = sims

    def _noise_bb(self, eq: float, opts: Options, bb: float) -> float:
        """Hur mycket EV-siffrorna kan vippa enbart pa Monte Carlo-slumpen.

        Equity ar ett stickprov, inte ett facit. Utan den har marginalen skulle
        samma beslut kunna fa betyget "ratt" ena gangen och "misstag" nasta —
        och en coach som motsager sig sjalv gar inte att lita pa.
        """
        if bb <= 0 or self.sims <= 0:
            return 0.0
        std_err = math.sqrt(max(eq * (1.0 - eq), 1e-6) / self.sims)
        scale = 2.0 * opts.pot + opts.call_amount
        return 2.0 * std_err * scale / bb

    # ---------- huvudingang ----------

    def review(
        self,
        hand: Hand,
        hero: Player,
        opts: Options,
        chosen_kind: str,
        chosen_to: float,
        villain_profile=None,
    ) -> Feedback:
        """Bedom hjaltens handling. Anropas direkt efter beslutet."""
        ctx = hand.context_for(hero)
        n_opp = max(1, sum(1 for p in hand.players if p.active) - 1)
        eq = calc_equity(hero.hole, hand.board, n_opp, sims=self.sims)
        ctx["_equity"] = eq

        rec = self.engine.analyze(ctx, villain_profile)

        if hand.street == "preflop":
            return self._review_preflop(hand, hero, opts, chosen_kind, chosen_to, eq, rec)
        return self._review_postflop(hand, hero, opts, chosen_kind, chosen_to, eq, rec)

    # ---------- preflop: charts ----------

    def _review_preflop(self, hand, hero, opts, chosen_kind, chosen_to, eq, rec) -> Feedback:
        notation = hand_notation(*hero.hole)
        pos = hero.position
        rec_action = rec.action

        chosen_class = self._action_class(chosen_kind, opts)
        rec_class = self._action_class(rec_action, opts)

        # Chartsen kan rekommendera en hojning nar ingen hojning ar mojlig
        if rec_class == "aggressive" and not opts.can_raise:
            rec_class = "passive"

        if chosen_class == rec_class:
            verdict = "ratt"
            headline = "Rätt beslut"
        elif {chosen_class, rec_class} == {"aggressive", "passive"}:
            verdict = "narapa"
            headline = "Nästan — rätt att fortsätta, fel växel"
        else:
            verdict = "misstag"
            headline = "Misstag"

        best_label = self._label_for(rec_action, rec.amount, hand.bb, opts)
        chosen_label = self._label_for(chosen_kind, chosen_to, hand.bb, opts)

        lines = [
            f"Hand: {notation} från {pos}",
            f"Equity mot {max(1, sum(1 for p in hand.players if p.active) - 1)} "
            f"motståndare: {eq:.0%}",
        ]
        if opts.call_amount > 0:
            pot_odds = opts.call_amount / (opts.pot + opts.call_amount)
            lines.append(
                f"Att syna kostar {_fmt(opts.call_amount, hand.bb)} "
                f"i en pott på {_fmt(opts.pot, hand.bb)} — du behöver {pot_odds:.0%}"
            )

        concept, category = self._preflop_concept(
            verdict, chosen_class, rec_class, notation, pos
        )
        loss_bb = self._preflop_cost(verdict, chosen_class, rec_class, opts, hand.bb)

        return Feedback(
            verdict=verdict,
            headline=headline,
            chosen_label=chosen_label,
            best_label=best_label,
            explanation=rec.reasoning,
            equity=eq,
            ev_loss_bb=loss_bb,
            pot_odds=(opts.call_amount / (opts.pot + opts.call_amount))
            if opts.call_amount > 0 else 0.0,
            lines=lines,
            concept=concept,
            category=category,
            street="preflop",
        )

    def _preflop_cost(self, verdict, chosen_class, rec_class, opts: Options,
                      bb: float) -> float:
        """Grov kostnad for ett preflop-fel, i big blinds.

        Preflop bedoms mot charts och inte mot EV, sa det finns ingen exakt
        siffra att hamta. Men utan nagon kostnad alls hamnar preflop-lackor
        alltid sist i rapportens rangordning — och att spela for manga hander
        ar den dyraste lackan som finns. Uppskattningen: att betala in i en
        pott man borde lamnat kostar ungefar halva insatsen i langden.
        """
        if verdict == "ratt" or bb <= 0:
            return 0.0
        invested = max(opts.call_amount, bb) / bb
        if rec_class == "fold" and chosen_class != "fold":
            return 0.5 * invested
        if rec_class != "fold" and chosen_class == "fold":
            return 0.5
        return 0.3

    def _preflop_concept(self, verdict, chosen_class, rec_class, notation, pos):
        if verdict == "ratt":
            return "", ""
        if rec_class == "fold" and chosen_class != "fold":
            return (
                f"{notation} ligger utanför den lönsamma rangen från {pos}. "
                "Att spela för många händer ur tidig position är den vanligaste "
                "och dyraste läckan som finns.",
                "Preflop: spelar för många händer",
            )
        if rec_class != "fold" and chosen_class == "fold":
            return (
                f"{notation} är stark nog att spela från {pos}. "
                "Att folda den ger bort chips du hade rätt till.",
                "Preflop: foldar för mycket",
            )
        if rec_class == "aggressive" and chosen_class == "passive":
            return (
                "Att bara syna här ger bort initiativet. Med en hand som är värd "
                "att spela vill du oftast höja: du vinner potten direkt ibland, "
                "och bygger den när du är bäst.",
                "Preflop: för passiv med spelbara händer",
            )
        return (
            "Att höja med den här handen bygger en pott du sällan är favorit i. "
            "Syn och se en billig flopp istället.",
            "Preflop: höjer för lätt",
        )

    # ---------- postflop: EV ----------

    def _review_postflop(self, hand, hero, opts, chosen_kind, chosen_to, eq, rec) -> Feedback:
        evs = self.ev_table(hand, hero, opts, eq)
        chosen_key = self._ev_key(chosen_kind, chosen_to, opts)
        tolerance = CLOSE_ENOUGH_BB + self._noise_bb(eq, opts, hand.bb)

        best_key = self._robust_best(evs, tolerance * hand.bb)
        best_ev = evs[best_key]
        chosen_ev = evs.get(chosen_key)
        if chosen_ev is None:
            chosen_ev = evs.get(self._nearest_bet_key(chosen_to, opts), best_ev)

        loss_bb = max(0.0, (best_ev - chosen_ev) / hand.bb)

        # Fel *handling* ar ett riktigt fel. Fel *storlek* pa ratt handling ar en
        # nyans — den ska papekas, men inte domas som ett misstag.
        passive = ("fold", "check", "call")
        sizing_only = chosen_key not in passive and best_key not in passive

        if loss_bb <= tolerance:
            # Inom felmarginalen — da ar valet lika bra som alternativet,
            # och vi ska inte lara ut en "battre" linje som inte ar battre.
            verdict, headline = "ratt", "Rätt beslut"
            loss_bb = 0.0
            best_key = chosen_key
        elif loss_bb <= MINOR_MISTAKE_BB + tolerance or sizing_only:
            verdict, headline = "narapa", "Nästan — men det finns bättre"
        else:
            verdict, headline = "misstag", "Misstag"

        pot_odds = (
            opts.call_amount / (opts.pot + opts.call_amount)
            if opts.call_amount > 0 else 0.0
        )

        lines = [
            f"Din equity: {eq:.0%} mot "
            f"{max(1, sum(1 for p in hand.players if p.active) - 1)} motståndare",
        ]
        if opts.call_amount > 0:
            lines.append(
                f"Pot odds: {_fmt(opts.call_amount, hand.bb)} att syna i en pott "
                f"på {_fmt(opts.pot, hand.bb)} — du behöver {pot_odds:.0%} för att gå plus"
            )
        lines.append("EV per alternativ (uppskattat, i bb):")
        for key in sorted(evs, key=lambda k: -evs[k]):
            marker = "  <- ditt val" if key == chosen_key else ""
            best_mark = "  <- bäst" if key == best_key and key != chosen_key else ""
            lines.append(f"   {key:<22} {evs[key] / hand.bb:+6.2f}{marker}{best_mark}")

        concept, category = self._postflop_concept(
            verdict, chosen_key, best_key, eq, pot_odds, opts
        )

        return Feedback(
            verdict=verdict,
            headline=headline,
            chosen_label=self._label_for(chosen_kind, chosen_to, hand.bb, opts),
            best_label=best_key,
            explanation=rec.reasoning,
            equity=eq,
            pot_odds=pot_odds,
            ev_loss_bb=loss_bb,
            lines=lines,
            concept=concept,
            category=category,
            street=hand.street,
        )

    def _postflop_concept(self, verdict, chosen_key, best_key, eq, pot_odds, opts):
        if verdict == "ratt":
            return "", ""
        if chosen_key == "fold" and opts.call_amount > 0 and eq > pot_odds:
            return (
                f"Du hade {eq:.0%} equity men behövde bara {pot_odds:.0%} för att "
                "synen skulle gå plus. Att folda när du får rätt odds ger bort pengar — "
                "du behöver inte vara favorit, bara tillräckligt ofta bäst.",
                "Postflop: foldar trots rätt pot odds",
            )
        if best_key == "fold":
            return (
                f"Med {eq:.0%} equity och krav på {pot_odds:.0%} förlorar du chips "
                "varje gång du betalar här. Fold är inte svaghet, det är att sluta "
                "betala för en hand du oftast förlorar.",
                "Postflop: betalar utan odds",
            )
        if chosen_key == "check" and best_key.startswith("bet"):
            return (
                f"Med {eq:.0%} equity är du oftast bäst — då vill du bygga potten. "
                "Att checka starka händer sparar inte pengar, det förlorar värde.",
                "Postflop: missar värdesatsningar",
            )
        if chosen_key.startswith("bet") and best_key == "check":
            return (
                "Att satsa här bygger en pott du sällan vinner. När handen är för "
                "svag för värde men för bra för att bluffa bort är check rätt.",
                "Postflop: satsar för ofta utan hand",
            )
        return (
            "Rätt riktning men fel storlek. Bet-storleken styr både hur ofta "
            "motståndaren foldar och hur mycket du vinner när du är bäst.",
            "Postflop: bet-storlek",
        )

    # ---------- EV-modell ----------

    def ev_table(self, hand: Hand, hero: Player, opts: Options, eq: float) -> Dict[str, float]:
        """EV for varje rimligt alternativ, i chips, relativt att folda nu."""
        pot = opts.pot
        to_call = opts.call_amount
        n_opp = max(1, sum(1 for p in hand.players if p.active) - 1)
        evs: Dict[str, float] = {}

        if to_call > 1e-9:
            evs["fold"] = 0.0
            # Syna: vi betalar to_call och spelar om hela potten
            evs["call"] = eq * (pot + to_call) - to_call
        else:
            # Gratis kort — vi ar med i potten som den ar
            evs["check"] = eq * pot

        if opts.can_raise:
            for fraction, name in ((0.5, "bet 1/2 pott"), (0.75, "bet 3/4 pott"), (1.0, "bet pott")):
                size = pot * fraction
                to_amount = opts.current_bet + size
                to_amount = max(to_amount, opts.min_raise_to)
                to_amount = min(to_amount, opts.max_raise_to)
                invest = to_amount - hero.street_bet
                if invest <= to_call + 1e-9:
                    continue
                label = name if to_call <= 1e-9 else name.replace("bet", "höj")
                evs[label] = self._ev_aggressive(pot, invest, to_call, eq, n_opp)

        return evs

    def _ev_aggressive(self, pot: float, invest: float, to_call: float,
                       eq: float, n_opp: int) -> float:
        """EV av att satsa/hoja `invest` chips i en pott pa `pot`.

        Tva utfall: motstandarna foldar (vi tar potten), eller nagon fortsatter
        (vi spelar om en pott som vuxit med bada insatserna).
        """
        fold_equity = self._fold_equity(pot, invest, n_opp, facing_bet=to_call > 0)
        called_pot = pot + invest  # motstandaren matchar ungefar var insats

        # Ju storre insats, desto starkare ar rangen som fortsatter mot oss.
        # Utan den har justeringen skulle modellen alltid rekommendera storsta
        # mojliga satsning sa fort vi har lite equity — vilket ar fel lardom.
        ratio = min(2.0, invest / pot) if pot > 1e-9 else 1.0
        eq_when_called = eq * max(0.55, 1.0 - 0.14 * ratio)

        ev_called = eq_when_called * (called_pot + invest) - invest
        return fold_equity * pot + (1 - fold_equity) * ev_called

    def _robust_best(self, evs: Dict[str, float], band_chips: float) -> str:
        """Valj det basta alternativet — men bland likvardiga, ta det trygga.

        Nar en bluffhojning och en fold ar statistiskt oskiljbara ska coachen
        inte saga "du borde ha bluffat". Modellens fold equity ar en gissning,
        och att lara ut marginella bluffar pa dess svagaste antagande ar
        precis fel sak att traina in.
        """
        top = max(evs.values())
        candidates = [k for k, v in evs.items() if v >= top - band_chips]
        rank = {"fold": 0, "check": 1, "call": 2}
        return min(candidates, key=lambda k: (rank.get(k, 3), -evs[k]))

    def _fold_equity(self, pot: float, invest: float, n_opp: int, facing_bet: bool) -> float:
        """Grov uppskattning av hur ofta alla foldar mot var insats."""
        if pot <= 1e-9:
            return 0.3
        ratio = min(2.0, invest / pot)
        single = min(0.65, 0.25 + 0.30 * ratio)
        if facing_bet:
            single *= 0.7   # nagon har redan visat styrka
        return max(0.05, single ** n_opp)

    # ---------- hjalpare ----------

    def _action_class(self, kind: str, opts: Options) -> str:
        if kind == "fold":
            return "fold"
        if kind in ("check", "call"):
            return "passive"
        return "aggressive"

    def _ev_key(self, kind: str, to_amount: float, opts: Options) -> str:
        if kind == "fold":
            return "fold"
        if kind == "check":
            return "check"
        if kind == "call":
            return "call"
        return self._nearest_bet_key(to_amount, opts)

    def _nearest_bet_key(self, to_amount: float, opts: Options) -> str:
        """Mappa en godtycklig hojning till narmaste alternativ i EV-tabellen."""
        size = max(0.0, to_amount - opts.current_bet)
        pot = max(opts.pot, 1e-9)
        fraction = size / pot
        prefix = "bet" if opts.call_amount <= 1e-9 else "höj"
        if fraction < 0.625:
            return f"{prefix} 1/2 pott"
        if fraction < 0.875:
            return f"{prefix} 3/4 pott"
        return f"{prefix} pott"

    def _label_for(self, kind: str, amount: float, bb: float, opts: Options) -> str:
        if kind == "fold":
            return "Fold"
        if kind == "check":
            return "Check"
        if kind == "call":
            return f"Syn {_fmt(opts.call_amount, bb)}"
        if kind == "all_in":
            return "All-in"
        verb = "Bet" if opts.call_amount <= 1e-9 else "Höj till"
        return f"{verb} {_fmt(amount, bb)}"
