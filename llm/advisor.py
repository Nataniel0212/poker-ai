"""
LLM integration — uses a local Ollama model to synthesize all analysis
into clear, actionable poker advice.
"""

from typing import Optional
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    ollama = None


SYSTEM_PROMPT = """Du är en expert-pokercoach som ger korta, tydliga råd i realtid.

Du får:
- Spelarens kort och boardet
- Equity-beräkningar
- GTO-rekommendation från strategy engine
- Motståndarens profil och exploits

Ditt svar ska vara:
1. **ACTION**: Exakt vad spelaren ska göra (FOLD / CHECK / CALL / BET / RAISE) och belopp
2. **CONFIDENCE**: 0-100%
3. **REASONING**: 1-2 meningar om varför
4. **IF_RAISED**: Vad göra om motståndaren raisar
5. **IF_CALLED**: Plan för nästa street
6. **EXPLOIT**: Eventuell exploit som används

Svara ALLTID i detta JSON-format:
{
    "action": "BET",
    "amount": 8.25,
    "confidence": 85,
    "reasoning": "Value bet med top pair top kicker, motståndaren foldar c-bets 70%",
    "if_raised": "Call - vi har 62% equity",
    "if_called": "Bet turn om blank, check om scary card",
    "exploit": "C-bet bluff vs fold_to_cbet 70%"
}

Tänk som en proffsspelare. Var koncis. Ingen fluff."""


@dataclass
class AdvisorConfig:
    model_name: str = "poker-ai"  # Ollama model name (custom poker model)
    temperature: float = 0.3     # Low temp for consistent decisions
    max_tokens: int = 300
    timeout: float = 5.0         # Max seconds to wait for response


class PokerAdvisor:
    """Connects to Ollama to get AI-powered poker advice."""

    def __init__(self, config: AdvisorConfig = None):
        self.config = config or AdvisorConfig()
        self._available = ollama is not None
        if not self._available:
            print("Warning: ollama package not installed. LLM advice unavailable.")
            print("Install with: pip install ollama")

    def is_available(self) -> bool:
        """Check if LLM is available and model is loaded."""
        if not self._available:
            return False
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if hasattr(models, 'models') else []
            return any(self.config.model_name in name for name in model_names)
        except Exception as e:
            print(f"Ollama check failed: {e}")
            return False

    def get_advice(self, game_context: dict, strategy_recommendation: dict,
                   opponent_profile: dict = None) -> dict:
        """Get AI advice for the current situation.

        Args:
            game_context: From GameState.get_context_summary()
            strategy_recommendation: From StrategyEngine.analyze() as dict
            opponent_profile: Opponent stats dict (optional)

        Returns:
            Dict with action, amount, confidence, reasoning, etc.
        """
        prompt = self._build_prompt(game_context, strategy_recommendation, opponent_profile)

        if not self._available:
            # Fallback: just return the strategy engine's recommendation
            return strategy_recommendation

        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )

            parsed = self._parse_response(response["message"]["content"], strategy_recommendation)
            # Normalize action to lowercase for consistency with strategy engine
            if "action" in parsed and isinstance(parsed["action"], str):
                parsed["action"] = parsed["action"].lower()
            return parsed

        except Exception as e:
            print(f"LLM error: {e}")
            return strategy_recommendation

    def _build_prompt(self, ctx: dict, rec: dict, opp: dict = None) -> str:
        """Build the prompt for the LLM."""
        parts = []

        # Current hand info
        parts.append(f"Mina kort: {ctx.get('hero_cards', '?')}")
        parts.append(f"Board: {ctx.get('community_cards', []) or 'Preflop'}")
        parts.append(f"Pot: ${ctx.get('pot', 0):.2f}")
        parts.append(f"Min stack: ${ctx.get('hero_stack', 0):.2f}")
        parts.append(f"Position: {ctx.get('hero_position', '?')}")
        parts.append(f"Street: {ctx.get('street', 'preflop')}")
        parts.append(f"Big blind: ${ctx.get('big_blind', 0):.2f}")

        # Opponents
        villains = ctx.get("villains", [])
        if villains:
            parts.append(f"\nMotståndare ({len(villains)} aktiva):")
            for v in villains:
                parts.append(f"  - {v.get('name', '?')} ({v.get('position', '?')}): "
                           f"stack ${v.get('stack', 0):.2f}, bet ${v.get('current_bet', 0):.2f}")

        # Actions this hand
        actions = ctx.get("all_actions", [])
        if actions:
            parts.append(f"\nAktioner denna hand:")
            for a in actions:
                parts.append(f"  [{a.get('street')}] {a.get('player')} {a.get('action')} ${a.get('amount', 0):.2f}")

        # Strategy engine recommendation
        parts.append(f"\n--- STRATEGY ENGINE ---")
        parts.append(f"Rekommendation: {rec.get('action', '?')} ${rec.get('amount', 0):.2f}")
        parts.append(f"Equity: {rec.get('equity', 0):.1%}")
        parts.append(f"Pot odds: {rec.get('pot_odds', 0):.1%}")
        parts.append(f"EV: ${rec.get('ev', 0):.2f}")
        parts.append(f"Reasoning: {rec.get('reasoning', '')}")

        if rec.get("exploit_note"):
            parts.append(f"Exploit: {rec['exploit_note']}")

        # Opponent profile
        if opp:
            parts.append(f"\n--- MOTSTÅNDARPROFIL ---")
            parts.append(f"Typ: {opp.get('player_type', 'unknown')}")
            parts.append(f"Händer: {opp.get('hands_played', 0)}")
            parts.append(f"VPIP: {opp.get('vpip', 0):.1f}%")
            parts.append(f"PFR: {opp.get('pfr', 0):.1f}%")
            parts.append(f"AF: {opp.get('af', 0):.1f}")
            parts.append(f"Fold to C-Bet: {opp.get('fold_to_cbet', 0):.1f}%")
            parts.append(f"Fold to 3-Bet: {opp.get('fold_to_three_bet', 0):.1f}%")
            parts.append(f"WTSD: {opp.get('wtsd', 0):.1f}%")

            tips = opp.get("exploit_tips", [])
            if tips:
                parts.append("Exploit-tips:")
                for tip in tips:
                    parts.append(f"  - {tip}")

        parts.append("\nGe ditt råd i JSON-format:")

        return "\n".join(parts)

    def _parse_response(self, text: str, fallback: dict) -> dict:
        """Parse LLM response into structured recommendation."""
        import json

        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # If JSON parsing fails, try to extract key info
        result = dict(fallback)
        text_lower = text.lower()

        for action in ["fold", "check", "call", "bet", "raise"]:
            if action in text_lower:
                result["action"] = action  # Keep lowercase to match strategy engine
                break

        return result

    def get_quick_advice(self, hero_cards: list, community_cards: list,
                         pot: float, position: str) -> str:
        """Get quick text advice without full game state.
        Useful for simple scenarios."""
        prompt = (
            f"Kort: {hero_cards}, Board: {community_cards or 'Preflop'}, "
            f"Pot: ${pot}, Position: {position}. "
            f"Vad bör jag göra? Kort svar."
        )

        if not self._available:
            return "LLM ej tillgänglig — använd strategy engine"

        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3, "num_predict": 150},
            )
            return response["message"]["content"]
        except Exception as e:
            return f"LLM-fel: {e}"
