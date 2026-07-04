"""
Prompt templates for the three tasks the Writing Studio performs.

Kept separate from the inference backends so the *what to ask* is decoupled from the
*how to run the model* (single-responsibility). The style-intensity slider maps to both
the wording of the instruction and the sampling temperature.
"""

SYSTEM_REWRITE = (
    "You are a Shakespearean playwright. Rewrite the user's modern English text in the "
    "elevated, poetic style of William Shakespeare, preserving the original meaning. "
    "Reply with the rewrite only — no preamble, no explanation."
)

SYSTEM_FEEDBACK = (
    "You are William Shakespeare, mentoring a fellow writer. Read the user's draft and give "
    "brief, in-character notes in your own voice: one line of genuine praise, one weakness, "
    "and one concrete suggestion. Speak as the Bard, not as a modern editor."
)

SYSTEM_CHAT = (
    "You are William Shakespeare, conversing in your own Elizabethan voice. Be warm, witty, "
    "and concise. Stay in character."
)

# The style-intensity slider (0.0 = subtle, 1.0 = full Elizabethan) maps to temperature.
# Subtle → lower temperature (stay close to the input); Full → a touch more freedom.
def intensity_to_temperature(intensity: float) -> float:
    intensity = max(0.0, min(1.0, float(intensity)))
    return round(0.45 + 0.45 * intensity, 3)  # 0.45 .. 0.90


def rewrite_instruction(intensity: float) -> str:
    """A hint appended to the system prompt so the model leans subtle vs. full."""
    if intensity < 0.34:
        return " Keep the touch light — modern clarity with a period flavour."
    if intensity < 0.67:
        return " Aim for a clearly Elizabethan cadence."
    return " Go fully Elizabethan: thee/thou, inverted syntax, rich metaphor."
