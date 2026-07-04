# VIEWS.md — The Private Shakespearean Writing Studio (Act III)

This file verbally describes **every view** in the Act III application, so Figma mocks can be
built from it and approved *before any UI code is written*.

## Product in one sentence

A local, **fully-offline** desktop-style web app (served by FastAPI on `localhost`, opened in the
browser) where a writer pastes their own draft and the on-device fine-tuned model rewrites it in
Shakespeare's voice or critiques it in-character — with a **provable, always-visible guarantee
that nothing leaves the machine**.

## Design principles (constraints the mocks must honor)

- **Offline-first, provably.** A persistent privacy indicator is visible in *every* view. No view
  may depend on any network asset — all fonts, icons, CSS, and JS are bundled locally (an external
  request would both break offline use and violate the no-egress promise).
- **Two-pane writing metaphor.** The core interaction is always "my draft" (left) → "the Bard's
  version" (right). Everything else is secondary.
- **Calm, paper-and-ink aesthetic.** Warm off-white "parchment" background, a serif display face
  for Shakespearean output, a plain sans for UI chrome and the user's own modern text. Readable,
  high-contrast, no clutter.
- **Local + single-user.** No login, no accounts, no cloud — those concepts do not appear anywhere.

---

## 1. App shell (persistent across all views)

Present in every view:

- **Top bar:** app title/wordmark ("Bardbox — Writing Studio") on the left; a **Privacy badge** on
  the right that reads **"🔒 Offline · 0 network calls"** in a calm green state. Clicking it opens
  the Privacy view (view 6). The badge is the emotional anchor of the whole product.
- **Left nav (slim, icon + label):** Studio, Feedback, Chat, Privacy, Settings.
- **Model chip:** small text showing the loaded model + quantization (e.g. "Qwen2.5-0.5B · Q4_K_M")
  and a status dot (green = loaded, grey = none loaded).

## 2. Studio view (primary / default)

The main workspace. A two-pane layout under the shell:

- **Left pane — "Your draft":** a large editable text area where the user types or pastes their own
  modern-English writing. Placeholder text invites a paste. A live character/word count sits below.
- **Center controls (between or above the panes):**
  - Primary button **"Rewrite in the Bard's voice"**.
  - A **style-intensity slider** (Subtle ↔ Full Elizabethan) mapped to generation temperature.
  - A **"Ground in the works"** toggle (optional RAG): when on, the rewrite may weave in or cite a
    relevant line from the actual corpus, shown with its source.
- **Right pane — "In the Bard's voice":** the model's rewrite, rendered in the serif display face on
  a subtly distinct parchment panel. Tokens **stream in** as generated. Below it: **Copy** and
  **"Rewrite again"** (re-roll) buttons. If grounding is on and a quote was used, a small
  **citation chip** (e.g. "— Hamlet, I.iii") appears.
- **Empty state (no output yet):** right pane shows a faint quill illustration and the hint
  "Your draft, transfigured, appears here."

### Studio sub-states
- **Generating:** right pane shows streaming text with a soft pulsing cursor; the Rewrite button
  becomes a **Stop** button.
- **No model loaded:** the center button is disabled and a banner reads "No model loaded — point
  Bardbox at a `.gguf` file in Settings," linking to the Settings view.
- **Generation error:** an inline, dismissible error card in the right pane ("The model stumbled —
  try again"), never a modal.

## 3. Feedback view

Same two-pane frame, but the right pane is **in-character critique** rather than a rewrite:

- Left pane: the user's draft (shared with Studio).
- Primary button **"Ask the Bard for notes"**.
- Right pane: a short, structured critique *written as Shakespeare mentoring a fellow writer* —
  e.g. praise, one weakness, one concrete suggestion — in his voice. Streams in like the rewrite.
- The critique is clearly framed as opinion/guidance, visually distinct from a rewrite (e.g. a
  "Notes from the Bard" header and a quotation-mark motif).

## 4. Chat view

A freeform conversation with the Bard persona, fully offline:

- Standard chat transcript (user bubbles in sans, Bard replies in serif on parchment).
- A message composer at the bottom with a Send button; replies stream token-by-token.
- A **"Clear conversation"** action. History lives only in memory / local storage — never sent out.
- Empty state: a one-line greeting in-voice ("Well met. What wouldst thou discuss?").

## 5. Settings view

Local configuration only:

- **Model:** the path to the loaded `.gguf` file, a **"Choose model file…"** control, and the
  detected model name + quantization. A note that models are optional/swappable.
- **Generation:** sliders/inputs for temperature, top-p, and max new tokens, with sensible defaults.
- **Grounding (RAG):** toggle to enable corpus grounding + a note that the index is built locally
  from the bundled complete-works text.
- **Everything on this screen is local;** copy explicitly states no settings are synced anywhere.

## 6. Privacy view (the signature view)

The view that makes the privacy claim *demonstrable*, not just asserted:

- **Big status line:** "🔒 This app is running entirely on your machine."
- **Live network panel:** a list of outbound network connections made by the app process, expected
  to read **"0 outbound connections"** with a green check. (Backed by the no-egress test.)
- **"Run privacy self-test" button:** triggers the bundled no-egress check and shows a pass/fail
  result inline (e.g. "✓ No egress detected over the last run").
- **Explainer copy:** a short, plain-language paragraph on *why* this matters for a writer —
  unpublished drafts are sensitive IP that should never be handed to a cloud API — plus an
  **airplane-mode challenge**: "Turn off Wi-Fi and keep working. Nothing changes. That's the point."
- **Failure state:** if the self-test ever detects an outbound connection, the badge across the app
  flips to a red "⚠ Network activity detected" state and this view explains what was seen.

## 7. First-run / onboarding state

On first launch with no model configured:

- A centered, single-column welcome explaining the three acts briefly and that this app is Act III.
- One clear call to action: **"Load a model to begin"** → Settings view.
- The privacy badge is already present and green even here (the app has made no calls).

## 8. Global loading & error states

- **App booting:** a minimal centered wordmark + "Warming the quill…" while the backend/model load.
- **Backend unreachable:** a full-view message "Can't reach the local Bardbox server" with the exact
  local command to start it — framed as a local process issue, never a network/cloud error.
