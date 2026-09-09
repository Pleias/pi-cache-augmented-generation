# Adapting this repo to the Bangui digital library

Status: assessment, not a spec. Written 2026-09-09 after reading the two attachments to
Jérémy Lachal's mail of 2026-09-07 (`docs/bangui-use-case/`). No design approved yet.

Audience: internal (Pleias), input to the upcoming call with BSF.

---

## 1. What we were sent

**`bangui_Rapport.pdf`** — BSF diagnostic mission report, Université de Bangui,
28 Feb–6 Mar 2026, by Leyre Gil Pedromingo and Ermeline Jaggi, under the EU technical
assistance project to the MESRSIT. 35 pages: diagnosis, documentary policy
recommendations, three intervention axes, interview annexes, focus groups.

**`bangui_Bibliothèque.html`** — the POC Jérémy built with Claude. It is a **static mock**:
one hardcoded query ("La mécanisation agricole en Afrique subsaharienne"), one hardcoded
synthesis, five hardcoded references, a fake 2.1-second "thinking" screen. No backend, no
retrieval, no model. Treat it as a UX contract, not a prototype.

What the mock commits us to, screen by screen:

- natural-language need, not keywords: "Décrivez votre besoin en quelques mots"
- an assistant synthesis paragraph that frames the topic and orders the reading
- five references, each with a **"Pourquoi cette référence"** commentary, a pull quote,
  year, journal, publisher, and explicit provenance ("via Cairn")
- a rights distinction on every card: **Accès ouvert · PDF** versus **Sous droits ·
  consultation locale**, with the footer promise that licensed content is encrypted on the
  device, non-exportable, non-redistributable
- **Citer cette source**
- a permanent **Serveur local · hors-ligne** badge
- BSF brand system: Ambit, coral `#FF5B29`, beige `#F5EFE6`, ink `#102547`

---

## 2. The constraints that actually bind

From the report, in rough order of how much they change the build.

| Constraint | Value | Source |
|---|---|---|
| Campus internet | effectively none | diagnostic, all interviews |
| Mobile data cost | 2 500 FCFA / week, 11 000 FCFA / month | MESRSIT focal point |
| Students | ~13 000, one public university | MESRSIT focal point |
| Student equipment | smartphone dominant, few computers | focus groups, Reco 1 |
| Box concurrency | about a dozen simultaneous connections per Ideas Cube | report footnote 7 |
| Update cadence | new content every 8–10 months, via a connected trip | Axe 2 |
| Power | frequent cuts, solar wanted; Sciences relatively stable | ENS director, site visits |
| Staff | no library-science school in the country, no digital training | Axe 3 |
| Pilot scope | 2 boxes, 2 doctoral schools | teachers' focus group |
| Political deadline | at least 2 libraries open at the 2026-2027 rentrée | Rector |

Two findings from the report deserve to sit above the rest, because they contradict the mock.

**Teachers asked for course-prep material, not research articles.** Their words: the
difficulty is not accessing science for research, it is finding teaching support for first
years. They proposed roughly **70% licence-level, 30% research**. They also said students
fall back on Wikipedia and ChatGPT for lack of anything at their level. The mock answers a
master's-thesis query. Those are two different products, and the licence-level one is both
the larger need and the cheaper one to serve.

**The ENS Koonbook is the strongest evidence in the file, in both directions.** A BSF kit
installed in 2017, heavily used by teachers and students, enriched with local theses and
student work. It stopped working in 2025 and library attendance collapsed. It proves the
demand and it prices the maintenance risk.

---

## 3. Assets we already hold

**This repo.** A shippable offline container: llama-server plus FastAPI plus a single-file
static UI, Qwen2.5-1.5B-Instruct Q4_K_M baked in, amd64 and arm64, zero content data,
layered `/data` persistence with an atomic index, crash-recovered migrations, content packs
in and out, French-first UI with English and Arabic. Plus `bench/` and `docs/TECH_REPORT.md`:
measured evidence of what a small model does and does not do on cheap hardware.

**French Science Commons.** Our own corpus, and the single biggest change to the picture
since the report was written.

| | |
|---|---|
| Documents | 1 250 000 |
| Words | 15 694 707 465 |
| Research articles | ~1 200 000 |
| Doctoral theses | ~52 000 |
| Coverage | 2007–2026 |
| Sources | HAL, theses.fr, OpenAlex |
| Format | Markdown from `dots.ocr`, structure preserved |
| Metadata | author, DOI, discipline, licence, publication date |

Derived, unverified, to be checked against the real files: about 12 600 words per document
on average, so roughly 18 000 tokens and 80 KB of Markdown per document. A 50 000-document
disciplinary subset lands near 4 GB of text before compression. That fits an offline box
comfortably.

Action item: locate the internal copy, lakeFS suspected, and confirm the layout and the
licence field distribution. The Hugging Face release is the fallback.

**Cairn.** About 1 000 articles and journal issues, francophone SHS, granted gratis to BSF
for this project. High editorial prestige, exactly the disciplines the Lettres faculty asked
for, and the only part of the corpus with rights strings attached.

**BSF platform.** OLIP and Ideas Cube for delivery, Omeka for the catalogue. Our app has to
be installable alongside them, not instead of them.

---

## 4. Fit analysis

### What carries over

The shipping shape. One container, no internet, `/data` as the only writable state, atomic
index, no external assets in the front end, multi-arch CI, `CONTAINER.md` as the single
public doc, French default. All of it maps onto an OLIP box with no technician on call.

Cache-Augmented Generation survives in exactly one place: **read-this-article mode**. The
student picks a reference, that document's prefix sits in the KV cache, and they ask
questions answered from that text alone. It is the part a search box cannot give them, and
it is the direct answer to the teachers' complaint about ChatGPT.

### What breaks

**The unit of work changes.** Today: one document, one user, chat. Bangui: a corpus,
ranked search, five results with commentary. A corpus of this size never fits a KV prefix.
Retrieval has to be real search over metadata and full text, and CAG becomes stage two.

**Concurrency.** `--parallel 1` is a measured invariant in this repo, and the Ideas Cube
serves about a dozen readers.

| Quantity | Value |
|---|---|
| Decode, 1.5B Q4, Pi 4 | 9–12 tok/s |
| Prefill, same | 43–50 tok/s |
| Mock-shaped answer, synthesis plus five commentaries | ~1 500 tokens |
| Single-user latency for that answer | about 2.5 minutes |
| Twelfth reader in the queue | about 30 minutes |

The mock as specified is not servable on a Pi 4. Three levers: better hardware, precompute
everything that does not depend on the query, and cut generated text per query to roughly
200 tokens. We will need all three.

**Document length versus context budget.** The app caps a fiche at 12 000 tokens and
budgets history against 16 384. An average French Science Commons document is longer than
that. So read-mode cannot ingest arbitrary corpus documents whole; it needs a packaging-time
summary or section-level selection.

**Slot files do not scale to a corpus.** A saved KV slot is about 300 MB and is specific to
the model and the llama.cpp build. Precomputing slots for a curated shelf of a few dozen
documents is realistic; precomputing them for 50 000 is not. Read-mode therefore splits:
instant on a curated shelf, prefill-on-demand elsewhere.

**Export.** `GET /api/pack/export` streams all of `/data` as a tar. That is precisely what
the Cairn grant will forbid, and what the mock promises on screen. Export needs a
per-corpus flag, licensed content non-exportable, before any Cairn content touches a box.

---

## 5. Proposed shape

Principle: **do the expensive work at packaging time on a real machine, keep the box dumb.**

**Packaging time**, offline, on our hardware, per release of the content pack:

1. Select the subset. French Science Commons filtered by discipline against the report's
   priority list, by licence field, and by level. Cairn overlaid on top with its rights flag.
2. Per document, generate once: a fiche of 300 to 600 tokens, a level tag from licence to
   doctorate, discipline codes, and a key-claim line. This is the artefact the box reads at
   query time, never the full PDF.
3. Build a lexical index over titles, abstracts, fiches and metadata. BM25 first, because it
   is cheap, debuggable and needs no GPU. Embeddings only if retrieval quality demands it.
4. Precompute KV slot files for a curated shelf, the semester's core readings, so read-mode
   opens instantly for the documents that matter most.
5. Emit citations deterministically from metadata. No model in that path, ever.

**Query time**, on the box:

1. Lexical retrieval returns the candidate references. **The model never chooses a reference
   that retrieval did not return.** With a 1.5B model, fabricated bibliography is a
   near-certainty otherwise, and a fabricated reference in a library assistant is a
   project-ending failure.
2. The model writes a short synthesis and one justification line per reference, reading the
   precomputed fiches only. Budget roughly 200 generated tokens per query, not 1 500.
3. Read-mode, on demand: the chosen document's fiche or section enters the prefix, and the
   existing CAG chat runs unchanged.
4. Citation export is a metadata lookup.

Two front ends over the same index, matching the 70/30 split: a licence-level "j'ai un
exposé à préparer" mode over course-support material, and the research reference finder the
mock shows.

---

## 6. Rights and safety

- Licensed content is non-exportable and non-redistributable at the API level, not by
  convention. Per-corpus flag, enforced in export, in pack build, and in the UI.
- Whether generating summaries from Cairn full text is covered by the grant is a
  text-and-data-mining question their letter probably does not address. Settle it before we
  package, not after.
- French Science Commons carries a per-document licence field. Redistribution offline must
  filter on it. Open access on a website is not the same permission as shipping a copy on a
  box in Bangui.
- Every answer credits journal, publisher and platform, as the mock already does.
- No answer without a source present on the box. The failure mode we are being hired to fix
  is precisely students trusting ungrounded generation.

---

## 7. Hardware

The single decision that moves everything else. The report leaves it open, Axe 2 is
unfunded, and the numbers above are measured on a Pi 4.

| Option | Expectation | Note |
|---|---|---|
| Pi 4, 4 GB | current baseline | one reader at a time, minutes per answer |
| Pi 5, 8 GB | roughly 2× | untested by us |
| x86 mini PC, N100 class, 16 GB | several times faster, 3–4B models in reach | untested by us |
| Mini PC with usable iGPU or a small discrete GPU | order of magnitude | power budget versus solar |

Everything past the first row is an estimate. None of it goes in a proposal until it is
benchmarked on the box BSF actually intends to ship, with our container and our corpus.

---

## 8. Open questions for the call

1. **Which box.** CPU, RAM, architecture, storage, power source, and how many concurrent
   readers per box. This decides model size, latency, and whether the mock is servable.
2. **Cairn delivery.** Full text or metadata and abstracts only. Formats. And explicitly:
   are derived summaries and an index permitted.
3. **OpenAlex.** The mail names it as the base. Note that French Science Commons already
   incorporates OpenAlex references and adds full text in Markdown, which OpenAlex alone does
   not have. Worth proposing our corpus as the substrate rather than a raw OpenAlex subset.
4. **Who packages content.** The report puts curation on BSF under Axe 1, with Omeka and
   OLIP. We need their catalogue format as the ingest contract.
5. **Which product first.** The teachers' 70% licence-level ask, or the mock's research
   assistant. Doing both well in one pilot is unlikely.
6. **Timeline.** The report targets the 2026-2027 rentrée and the Rector wants two libraries
   open. That date has arrived. What ships this term.
7. **Local content.** The 1 316 theses since 1960 sit in an Excel file on the research
   director's personal drive, undigitised. Ingesting local production is the feature that
   makes the box theirs rather than ours. Is it in scope, and who scans.
8. **Who keeps it alive.** The Koonbook died in 2025 and nobody could revive it. Named
   referents, cascade training, and a spare-parts plan, or we repeat it.

---

## 9. Concrete consequences for this repo

- `bench/` stays as evidence and gains a concurrency benchmark on the target box.
- `app/` grows a retrieval layer and a `/api/search` endpoint; `/api/ask` splits into
  corpus-level answer and document-level chat.
- `/api/ingest` becomes catalogue import from a packaged corpus, alongside the existing
  single-fiche path for local theses.
- `/api/pack/export` gains rights enforcement.
- The 20-document library cap, the single slot, and the single `_slot_lock` all need
  revisiting under multi-reader load. The lock is currently correct for one user and wrong
  for twelve.
- Arabic can stay; French primary and English secondary are what the report asks for.

## 10. Out of scope, deliberately

Digitising the microfiches, recovering the Boukoko agricultural archive, Research4Life
credential management, hardware procurement, and librarian training. All real, all in the
report, none of them ours.
