"""Sajtprofiler — allt som ar specifikt for ett visst pokerspel.

En profil per sajt gor programmet sajtoberoende: koden vet ingenting om
247freepoker eller nagon annan sida. Den vet bara hur man laser ett hornindex.
Vad *just den har* sajtens glyfer ser ut som ligger i profilen, och den byggs
av kalibreringen.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from screenadvisor.glyphs import TemplateStore

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "screenadvisor", "profiles",
)


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "profil"


@dataclass
class Profile:
    name: str = "standard"
    region: Optional[Tuple[int, int, int, int]] = None      # x, y, w, h pa skarmen
    hero_zone: Optional[Tuple[int, int, int, int]] = None   # relativt region
    seat_zones: Optional[List[Tuple[int, int, int, int]]] = None  # motstandarnas satesrutor
    table_size: int = 5
    templates: TemplateStore = field(default_factory=TemplateStore)

    @property
    def slug(self) -> str:
        return _slug(self.name)

    @property
    def path(self) -> str:
        return os.path.join(PROFILE_DIR, f"{self.slug}.json")

    @property
    def default_opponents(self) -> int:
        return max(1, self.table_size - 1)

    # ---------- lagring ----------

    def save(self) -> str:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        payload = {
            "name": self.name,
            "region": list(self.region) if self.region else None,
            "hero_zone": list(self.hero_zone) if self.hero_zone else None,
            "seat_zones": ([list(z) for z in self.seat_zones]
                           if self.seat_zones else None),
            "table_size": self.table_size,
            "templates": self.templates.to_dict(),
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        return self.path

    @classmethod
    def load(cls, name: str) -> "Profile":
        path = os.path.join(PROFILE_DIR, f"{_slug(name)}.json")
        if not os.path.exists(path):
            return cls(name=name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, OSError):
            return cls(name=name)

        profile = cls(name=data.get("name", name))
        region = data.get("region")
        hero = data.get("hero_zone")
        seats = data.get("seat_zones")
        profile.region = tuple(region) if region else None
        profile.hero_zone = tuple(hero) if hero else None
        profile.seat_zones = [tuple(z) for z in seats] if seats else None
        profile.table_size = int(data.get("table_size", 5))
        profile.templates = TemplateStore.from_dict(data.get("templates") or {})
        return profile

    @classmethod
    def list_all(cls) -> List[str]:
        if not os.path.isdir(PROFILE_DIR):
            return []
        names = []
        for filename in sorted(os.listdir(PROFILE_DIR)):
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(PROFILE_DIR, filename), "r",
                          encoding="utf-8") as fh:
                    names.append(json.load(fh).get("name", filename[:-5]))
            except (ValueError, OSError):
                names.append(filename[:-5])
        return names

    def status(self) -> str:
        missing_ranks, missing_suits = self.templates.missing()
        if not self.region:
            return "Ingen skarmregion vald — kor kalibreringen"
        if missing_ranks or missing_suits:
            parts = []
            if missing_ranks:
                parts.append(f"rankar: {' '.join(missing_ranks)}")
            if missing_suits:
                parts.append(f"farger: {' '.join(missing_suits)}")
            return "Saknar " + ", ".join(parts)
        return "Komplett"
