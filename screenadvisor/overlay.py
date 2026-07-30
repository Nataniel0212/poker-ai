"""Radgivningsfonstret — ligger ovanpa spelet och uppdateras medan du spelar.

Trad-uppdelningen ar medvetet forsiktig: en vanlig Python-trad laser skarmen och
raknar, och lagger resultatet i en las-skyddad ruta. Qt-fonstret hamtar sedan
resultatet med en timer i sin egen trad. Ingen Qt-widget rors nagonsin fran
bakgrundstraden, vilket ar den vanligaste orsaken till svarfunna krascher.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from screenadvisor import capture
from screenadvisor.advice import Advice, build_advice
from screenadvisor.profile import Profile
from screenadvisor.reader import TableRead, read_table

SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RED = ("h", "d")


@dataclass
class Snapshot:
    read: Optional[TableRead] = None
    advice: Optional[Advice] = None
    error: str = ""
    elapsed_ms: int = 0


class AdvisorWorker(threading.Thread):
    """Laser skarmen i bakgrunden och halller senaste radet uppdaterat."""

    def __init__(self, profile: Profile, opponents: Optional[int] = None,
                 interval: float = 0.4, sims: int = 8000):
        super().__init__(daemon=True)
        self.profile = profile
        self.interval = interval
        self.sims = sims
        self._opponents = opponents or profile.default_opponents
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._cache_key = None
        self._cache_advice: Optional[Advice] = None

    # ---------- gransssnitt mot UI ----------

    @property
    def opponents(self) -> int:
        with self._lock:
            return self._opponents

    def set_opponents(self, value: int) -> None:
        with self._lock:
            self._opponents = max(1, min(9, value))
            self._cache_key = None      # tvinga omrakning

    def latest(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def stop(self) -> None:
        self._stop.set()

    # ---------- arbetslopet ----------

    def run(self) -> None:
        import os
        try:
            profile_mtime = os.path.getmtime(self.profile.path)
        except OSError:
            profile_mtime = 0.0

        while not self._stop.is_set():
            start = time.time()
            snapshot = Snapshot()
            try:
                # Ladda om profilen nar den andrats pa disk — da slar nya
                # inlarda glyfer igenom direkt utan att fonstret startas om.
                try:
                    mtime = os.path.getmtime(self.profile.path)
                except OSError:
                    mtime = profile_mtime
                if mtime != profile_mtime:
                    profile_mtime = mtime
                    self.profile = Profile.load(self.profile.name)
                    self._cache_key = None

                frame = capture.grab(self.profile.region)
                opponents = self.opponents
                read = read_table(
                    frame,
                    self.profile.templates,
                    hero_zone=self.profile.hero_zone,
                    opponents_override=opponents,
                )
                snapshot.read = read

                if read.usable:
                    # Rakna bara om nar situationen faktiskt andrats — equity
                    # ar det enda dyra steget, och kortet ligger stilla langa
                    # stunder medan motstandarna funderar.
                    key = (tuple(read.hero), tuple(read.board), opponents)
                    if key != self._cache_key:
                        self._cache_advice = build_advice(
                            read.hero, read.board, opponents, sims=self.sims
                        )
                        self._cache_key = key
                    snapshot.advice = self._cache_advice
            except Exception as exc:                    # noqa: BLE001
                snapshot.error = f"{type(exc).__name__}: {exc}"

            snapshot.elapsed_ms = int((time.time() - start) * 1000)
            with self._lock:
                self._snapshot = snapshot

            time.sleep(max(0.05, self.interval - (time.time() - start)))


# ---------- formatering ----------

def card_html(card: str, size: int = 22) -> str:
    if len(card) != 2:
        return card
    colour = "#e24a4a" if card[1] in RED else "#e8e8e8"
    return (f'<span style="color:{colour}; font-size:{size}px; font-weight:bold">'
            f'{card[0]}{SUIT_SYMBOL.get(card[1], card[1])}</span>')


def cards_html(cards, size: int = 22) -> str:
    if not cards:
        return '<span style="color:#777">—</span>'
    return " ".join(card_html(c, size) for c in cards)


def render_text(snapshot: Snapshot, opponents: int) -> str:
    """Ren text — anvands av konsollaget."""
    if snapshot.error:
        return f"FEL: {snapshot.error}"
    read = snapshot.read
    if read is None:
        return "Startar..."
    lines = [
        f"Dina kort: {' '.join(read.hero) if read.hero else '—'}",
        f"Bord:      {' '.join(read.board) if read.board else '—'}",
        f"Motstandare: {opponents}",
    ]
    if read.unknown_cards:
        lines.append(f"OKANDA KORT: {read.unknown_cards} — kor kalibreringen")
    if read.note:
        lines.append(read.note)
    if snapshot.advice:
        lines.append("")
        lines.append(snapshot.advice.headline)
        lines.extend("  " + l for l in snapshot.advice.lines)
    return "\n".join(lines)


# ---------- Qt-fonstret ----------

def run_overlay(profile: Profile, opponents: Optional[int] = None,
                sims: int = 8000, interval: float = 0.4) -> int:
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel,
                                     QPushButton, QVBoxLayout, QWidget)
    except ImportError:
        print("PyQt6 saknas — kor med --console istallet.")
        return 1

    worker = AdvisorWorker(profile, opponents, interval=interval, sims=sims)
    worker.start()

    app = QApplication([])

    window = QWidget()
    window.setWindowTitle(f"Pokerrad — {profile.name}")
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    window.resize(390, 460)
    window.setStyleSheet("background:#16181c; color:#e8e8e8;")

    # Stall fonstret UTANFOR lasregionen — hamnar det over bordet laser
    # skarmlasaren fonstrets egna pixlar istallet for korten.
    if profile.region is not None:
        rx, ry = profile.region[0], profile.region[1]
        window.move(max(0, rx - 390 - 12), max(0, ry))

    layout = QVBoxLayout(window)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)

    cards_label = QLabel()
    cards_label.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(cards_label)

    headline = QLabel()
    headline.setWordWrap(True)
    headline.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
    layout.addWidget(headline)

    detail = QLabel()
    detail.setWordWrap(True)
    detail.setFont(QFont("Consolas", 10))
    detail.setStyleSheet("color:#b9bcc4;")
    detail.setAlignment(Qt.AlignmentFlag.AlignTop)
    layout.addWidget(detail, 1)

    # Motstandarraknare
    controls = QHBoxLayout()
    opp_label = QLabel()
    minus = QPushButton("−")
    plus = QPushButton("+")
    for button in (minus, plus):
        button.setFixedWidth(34)
        button.setStyleSheet("background:#2a2e36; border:none; padding:5px;"
                            "font-size:15px; color:#e8e8e8;")
    controls.addWidget(QLabel("Motståndare kvar i handen:"))
    controls.addWidget(opp_label)
    controls.addStretch(1)
    controls.addWidget(minus)
    controls.addWidget(plus)
    layout.addLayout(controls)

    status = QLabel()
    status.setStyleSheet("color:#6d7280; font-size:10px;")
    layout.addWidget(status)

    minus.clicked.connect(lambda: worker.set_opponents(worker.opponents - 1))
    plus.clicked.connect(lambda: worker.set_opponents(worker.opponents + 1))

    def tick():
        snapshot = worker.latest()
        opp = worker.opponents
        opp_label.setText(f"<b>{opp}</b>")

        if snapshot.error:
            headline.setText("Fel vid läsning")
            headline.setStyleSheet("color:#e24a4a;")
            detail.setText(snapshot.error)
            return

        read = snapshot.read
        if read is None:
            headline.setText("Startar…")
            return

        cards_label.setText(
            f'<div style="margin-bottom:2px">Dina kort &nbsp; {cards_html(read.hero, 26)}</div>'
            f'<div>Bord &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {cards_html(read.board, 20)}</div>'
        )

        if read.unknown_cards:
            headline.setText(f"{read.unknown_cards} kort kunde inte läsas")
            headline.setStyleSheet("color:#e8a33d;")
            detail.setText(
                "Programmet gissar aldrig. Kör kalibreringen igen så det får\n"
                "lära sig glyferna som saknas:\n\n"
                f"    python watch.py --calibrate --profile \"{profile.name}\""
            )
        elif snapshot.advice and read.usable:
            advice = snapshot.advice
            colour = "#4ac26b" if advice.equity >= 0.55 else (
                "#e8a33d" if advice.equity >= 0.35 else "#e24a4a")
            headline.setText(advice.headline)
            headline.setStyleSheet(f"color:{colour};")
            detail.setText("\n".join(advice.lines))
        else:
            headline.setText("Väntar på kort")
            headline.setStyleSheet("color:#8a8f98;")
            detail.setText(read.note or "Inga kort syns i det valda området.")

        status.setText(f"läst på {snapshot.elapsed_ms} ms   ·   {profile.status()}")

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(250)

    window.show()
    code = app.exec()
    worker.stop()
    return code


def run_console(profile: Profile, opponents: Optional[int] = None,
                sims: int = 8000, interval: float = 0.5) -> int:
    """Textlage — for nar PyQt6 inte finns eller vid felsokning."""
    worker = AdvisorWorker(profile, opponents, interval=interval, sims=sims)
    worker.start()
    previous = None
    print(" Läser skärmen. Ctrl+C avslutar.")
    try:
        while True:
            snapshot = worker.latest()
            text = render_text(snapshot, worker.opponents)
            if text != previous:
                print("\n" + "=" * 52)
                print(text)
                previous = text
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
    finally:
        worker.stop()
    return 0
