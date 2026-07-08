# User stories — local-first CAG app

**Purpose:** product-side backlog suggesting NEW functionality on top of the shipped app.
**Grounding:** the product today is one Docker image (x86/arm64, Pi 4 floor) that ingests
**one document ≤12k tokens**, prefills its KV cache **once**, persists the slot + a
`history.json`, restores on restart, and serves multi-turn grounded chat over a single
static page + JSON/SSE API — **single user, single slot, fully offline**. See `README.md`,
`app/CONTAINER.md`, `docs/USE_CASES.md`, `docs/TECH_REPORT.md` §8.

**On-device reality every story below is checked against (from the benchmark):**
- Decode ~12 tok/s on a Pi and **falls as cached context grows** → the interactive unit is
  **one bounded document**, never a live multi-doc concatenation. Big corpora are batch-only.
- **One KV slot** in RAM (~2 GB peak on a 4 GB Pi). Switching documents means either
  re-prefill (~4–5 min for 12k tokens) **or** save/restore **slot files on disk** (seconds,
  ~300 MB per slot).
- Prefill 44–53 tok/s → a 12k-token ingest is a **one-time ~4–5 min** cost.
- **No cloud, no external calls, ever.** Teacher / low-literacy users can't write good queries.

Feasibility tags: **cheap** = static-page + FastAPI endpoint + reuse of the slot/history
file model; **moderate** = new persistent structure or multi-file orchestration on the
volume; **hard** = pushes RAM/decode budget or needs a second model.

---

## Teachers (offline / low-connectivity schools)

**US-1. A primary teacher wants a library of her lesson fiches so she can switch between today's lessons without re-pasting and re-waiting 5 minutes each.**
- Story: Madame Dubois teaches geography, sciences and history from paper fiches. In the morning she wants "le jour et la nuit" ready, then after recess "le cycle de l'eau" — today each switch means re-pasting the text and waiting out a full prefill.
- Suggests feature: a **multi-document library** — each ingested fiche keeps its own saved KV slot on disk; pick one from a list to make it active.
- Feasibility on the Pi: **moderate** — generalize `slot0.bin` to per-doc slot files (~300 MB each on disk), swap = restore in seconds; only one slot is ever live in RAM, so the decode wall is respected.
- Priority: **P1**

**US-2. A teacher wants one-tap "make 5 exercises / a quiz / a simpler re-explanation" from the current fiche so she preps a lesson without composing prompts.**
- Story: Prepping Sunday night, she wants differentiated exercises and a short assessment off the cached lesson, but she isn't going to write careful instructions for each — she wants buttons.
- Suggests feature: **lesson-prep prompt presets** — curated instruction templates fired as one-tap actions over the cached document.
- Feasibility on the Pi: **cheap** — pure prompt engineering; the document prefix stays cached, each preset is just a new turn, near-zero added prefill.
- Priority: **P1**

**US-3. A teacher wants to prepare all of next week's fiches in one Sunday-night batch so Monday morning every lesson loads instantly.**
- Story: She has ten fiches for the week. She'd rather ingest them all at once tonight — even if it's slow — than lose class time to prefills during the day.
- Suggests feature: a **background ingest queue** — paste several documents, they prefill one after another and each lands as a saved slot in the library.
- Feasibility on the Pi: **moderate** — prefill is a one-time ~4–5 min per doc; a serial queue over the single slot fits `--parallel 1`, no concurrency needed.
- Priority: **P2**

**US-4. A teacher wants to hand a colleague the prepared fiches on a USB stick so the whole grade level doesn't each re-ingest the same curriculum.**
- Story: Three teachers share the CE2 geography set. One prepares the fiches once; the others should get the ready-to-use library without repeating 30 minutes of prefills.
- Suggests feature: **export / import a content pack** — bundle selected documents + their slots + histories into one file to copy between devices.
- Feasibility on the Pi: **moderate** — slots are just files on the volume; a tar + manifest export/import reuses the existing file model entirely, no re-prefill on import.
- Priority: **P2**

**US-5. A French-speaking teacher wants the whole interface in French so she and her pupils can navigate it.**
- Story: The origin scenario is French primary school. English UI labels are a barrier for both the teacher and the children using it.
- Suggests feature: **UI localization** (French first) — a language toggle over the single static page.
- Feasibility on the Pi: **cheap** — static-page string table; no model or backend change.
- Priority: **P1**

**US-6. A teacher wants to lock the app to today's lesson so pupils using the class Pi can't swap the document or wipe the chat.**
- Story: The Pi sits at a back-of-class station. Pupils should ask questions about the loaded fiche, not re-ingest a random paste or hit "Clear chat".
- Suggests feature: a **"student mode" lock** — hides ingest/clear/model controls, leaving only ask-and-read.
- Feasibility on the Pi: **cheap** — a UI mode flag + guarding a couple of endpoints; no new state.
- Priority: **P2**

---

## Students (self-study)

**US-7. A student wants a "quiz me" mode over the chapter that asks questions and grades the answers so revision is active, not just reading.**
- Story: Before a test, Léa wants the Pi to interrogate her on the water cycle chapter — ask, wait for her answer, tell her if she's right and why — grounded in the whole chapter so it never tests something the material didn't cover.
- Suggests feature: **Socratic / quiz-me mode** — a preset that puts the model in the examiner role over the cached document.
- Feasibility on the Pi: **cheap** — prompt-engineering preset on the existing multi-turn loop; whole-context grounding is exactly CAG's regime.
- Priority: **P1**

**US-8. A stuck student wants to ask in vague, everyday words and still get help, because they can't formulate a precise question.**
- Story: "Je comprends pas le jour et la nuit." There is no precise chunk to retrieve — the student needs the model to work from the whole coherent lesson and a fuzzy prompt.
- Suggests feature: a **"help me understand" tutor preset** tuned to expand vague prompts against full context (plus example starter prompts on the page).
- Feasibility on the Pi: **cheap** — prompt preset + a few suggested-question chips; full-context CAG already handles the vague-query case that RAG can't.
- Priority: **P1**

**US-9. A student wants to star good answers to auto-build a revision sheet for the chapter.**
- Story: Over a study session Léa gets a few crisp explanations she wants to keep. She'd like to star them and export a clean one-page revision sheet at the end.
- Suggests feature: **bookmark-to-notes** — star an answer, append it to a per-document `notes.json`, export as a printable sheet.
- Feasibility on the Pi: **cheap** — mirrors the `history.json` atomic-append pattern; static-page export.
- Priority: **P2**

**US-10. A student self-studying several subjects wants each subject's chat to resume where it left off when they come back tomorrow.**
- Story: Léa revises maths tonight, geography tomorrow. Each subject should keep its own conversation so she picks up mid-thread, not from scratch.
- Suggests feature: **per-document persistent history** carried into the multi-doc library (each slot has its own `history.json`).
- Feasibility on the Pi: **cheap–moderate** — the single-doc history already persists and replays; extend the pattern per library entry (rides on US-1).
- Priority: **P2**

---

## Privacy-sensitive local users (lawyer / medic / researcher)

**US-11. A lawyer wants to load a confidential contract and Q&A it with a visible guarantee that nothing leaves the device.**
- Story: Reviewing a client NDA on a train with no signal, she needs to interrogate the clauses and be *sure* no text is sent anywhere.
- Suggests feature: an explicit **offline / no-network indicator** — a persistent badge asserting local-only operation (and, optionally, refusing to start if an outbound interface is detected).
- Feasibility on the Pi: **cheap** — the app already makes zero external calls; this surfaces and hardens that fact in the UI.
- Priority: **P2**

**US-12. A medic or researcher wants a one-click wipe of the document, cache and chat after reading a sensitive file, leaving no trace on the device.**
- Story: After summarizing a confidential patient dossier, the clinician wants to be certain the text, KV slot and conversation are all gone before handing the tablet back.
- Suggests feature: **one-click secure purge** — delete the active slot, its document and history atomically (and optionally the whole library).
- Feasibility on the Pi: **cheap** — file deletion on `/data` + reset slot; inverse of the existing ingest/persist flow.
- Priority: **P1**

**US-13. A researcher wants to keep several confidential documents loaded and switch between them, each with its own cache, without any cross-leakage.**
- Story: She's reading three separate confidential reports and wants to jump between them across the week without re-ingesting — and without one document's context bleeding into another's answers.
- Suggests feature: the **multi-document library** again, with strict per-slot isolation (only the selected slot is ever in the prompt).
- Feasibility on the Pi: **moderate** — same slot-file mechanism as US-1; isolation is inherent because only one slot is live at a time.
- Priority: **P2**

**US-14. A user on a shared device wants a local PIN so the loaded document and chat aren't exposed to whoever picks up the machine next.**
- Story: The Pi is a shared family / clinic-desk device. The confidential doc and its conversation shouldn't be visible without a code.
- Suggests feature: a **local PIN/passphrase gate** on the app (and on the persisted library).
- Feasibility on the Pi: **moderate** — a local auth layer over the static page + API; optional at-rest slot encryption is heavier and can wait.
- Priority: **P3**

---

## Operators / deployers (NGO tech staff imaging fleets of Pis)

**US-15. NGO tech staff want to image 50 Pis with the term's curriculum already prefilled so teachers open the app to ready-to-use lessons.**
- Story: A field team images 50 Pi 4s for rural schools. They want each device to arrive with the geography/science fiches already ingested — no teacher should sit through prefills on day one.
- Suggests feature: **pre-baked content packs** — seed the `/data` volume with prepared slots + histories + docs at imaging time (produced once, copied to every card).
- Feasibility on the Pi: **moderate** — slots are portable ~300 MB files; build the pack once, ship it in the image or on the SD image. Reuses US-4's bundle format.
- Priority: **P1**

**US-16. Operators want to push a curriculum update to the whole fleet by shipping a new pack, not by re-ingesting on every device.**
- Story: Mid-year the syllabus changes. The team wants to distribute an updated library bundle (over sneakernet / local Wi-Fi) that devices import in seconds, rather than re-prefilling on 50 Pis.
- Suggests feature: **versioned importable content packs** — import replaces/extends the library and records a version.
- Feasibility on the Pi: **moderate** — import unpacks slot files and registers them; no re-prefill, so it's seconds per device.
- Priority: **P2**

**US-17. Operators want a health/readiness endpoint to verify each imaged Pi loaded its model and cache correctly before handing devices out.**
- Story: Before shipping 50 units, QA wants a scriptable check per device: model present, expected slots restore, a quick decode self-test passes.
- Suggests feature: a **`/api/health` readiness endpoint** reporting model, restored slots, and a short tok/s self-test.
- Feasibility on the Pi: **cheap** — a read-only FastAPI endpoint over already-tracked state.
- Priority: **P2**

**US-18. Operators want to cap disk use so ~300 MB slots don't overrun small SD cards on cheap Pis.**
- Story: Devices ship with modest SD cards. A growing library of 300 MB slots plus histories could fill the card and brick the device.
- Suggests feature: a **library size cap with LRU eviction** of slot files (documents + history kept, slot re-derivable by re-prefill).
- Feasibility on the Pi: **moderate** — track slot sizes in the library index, evict least-recently-used slots past a configurable cap.
- Priority: **P3**

**US-19. A low-literacy or very young pupil wants to hear answers read aloud and ask by voice, because they can't yet read or type fluently.**
- Story: A 6-year-old at the class station can't reliably read the answer or type a question, but could speak and listen.
- Suggests feature: **offline text-to-speech (and optionally speech-to-text)** on device.
- Feasibility on the Pi: **hard** — needs a second model in RAM alongside the ~2 GB CAG footprint on a 4 GB Pi; small offline TTS (e.g. a Piper-class voice) is plausible, offline STT is heavier and competes with the KV budget. Prototype TTS-only on 8 GB first.
- Priority: **P3**

---

## Top 3 candidate features

These three resolve the most stories at once and stay inside the slot-file / `history.json`
/ single-static-page architecture — no GPU, no cloud, one slot live in RAM.

### 1. Document Library (multi-slot swap) — resolves US-1, US-3, US-10, US-13, US-15
Generalize the single `slot0.bin` into per-document slot files under `/data/slots/<id>.bin`,
indexed by a `library.json` that mirrors the `history.json` pattern (id → title, token count,
slot path, per-doc history path, last-used). Ingest writes a **new** slot instead of
overwriting; a UI picker makes one active by **restoring that slot (seconds) and loading its
history**. Only one slot is ever in the prompt, so the decode wall and ~2 GB RAM ceiling are
untouched — the cost is disk (~300 MB/slot), bounded later by US-18's LRU cap. This is the
backbone feature: it turns "one document" into "one active document out of many," which is
what teachers, multi-subject students, researchers and pre-loaded fleets all need.

### 2. Guided Modes / prompt presets — resolves US-2, US-7, US-8 (and helps US-6)
A small server-side set of curated instruction templates — *quiz-me / Socratic*, *explain
simpler*, *more examples*, *make exercises*, *help-me-understand (vague-prompt)* — surfaced as
one-tap buttons and suggested-question chips on the static page. Each preset is just a new turn
fired over the already-cached document prefix, so added prefill is near zero and there is **no
model or architecture change** — pure prompt engineering. This directly answers the
"teacher/low-literacy users can't write good queries" constraint and monetizes CAG's real edge
(cheap, repeated, whole-context work on one bounded document).

### 3. Content Packs — export / import / purge — resolves US-4, US-12, US-15, US-16
One file-level operation family over the volume. **Export** tars selected library entries
(docs + slots + histories + a versioned manifest); **import** unpacks them into `/data` and
registers them in `library.json` — seconds per device, no re-prefill, so operators seed and
update 50 Pis by sneakernet and teachers share a prepared curriculum on a USB stick. The
**purge** inverse deletes a slot + its document + history atomically for the privacy wipe. It
reuses the existing "everything persistent is a file on `/data`" model wholesale and is the
deployment and privacy story rolled into one primitive.
