# Internationalization Plan — CAG Chat (offline Raspberry Pi)

Scope: localize the single-file front (`app/server/static/index.html`) into **4 UI
languages** plus English, without touching the backend prompt/logic. The model baked in
is **Qwen2.5-1.5B-Instruct**, so the chosen UI languages are constrained to what that
model can actually *understand and generate* at the 1.5B scale — not just what the model
card lists.

---

## 1. Language selection

### The hard constraint: Qwen2.5-1.5B, not Qwen2.5-in-general

The Qwen2.5 model card advertises "over 29 languages." The DeepWiki enumeration groups
them as:

- **European:** English, French, Spanish, Portuguese, German, Italian, Russian, Dutch,
  Polish, Czech, Romanian, Ukrainian
- **East Asian:** Chinese (Simplified & Traditional), Japanese, Korean
- **Southeast Asian:** Vietnamese, Thai, Indonesian, Malay
- **South Asian:** Hindi, Bengali, Urdu
- **Middle Eastern:** Arabic, Persian, Turkish
- **African:** Swahili, Amharic
- **Others:** Greek, Hebrew

**But two facts override that list for a 1.5B model:**

1. **Qwen's own multilingual *instruction-following* benchmark (a translated IFEval) is
   run on exactly 8 languages: Arabic, Spanish, French, Indonesian, Japanese, Korean,
   Portuguese, Vietnamese.** This is the team's implicit "we trust generation quality
   here" shortlist — a much stronger signal than the marketing "29+" list. All four of
   our picks are inside it.
2. **Low-resource languages collapse at small scale.** Independent evaluation reports
   Qwen2.5-**7B** scoring ~0.00% language consistency and near-zero logical accuracy on
   Hausa, Swahili, Yoruba and Amharic. At 1.5B it is worse, and a *grounded-QA study
   tool that hallucinates in the child's own language is worse than no tool.* Generation
   quality also degrades on non-Latin scripts (Devanagari, Bengali) and drops sharply
   below the "big Latin-script + Arabic" tier.

> **Surprise worth flagging:** the origin scenario is francophone West Africa (Senegal
> CEB), yet the region's actual mother tongues — **Wolof, Bambara, Fula/Pulaar, Hausa** —
> are *unusable or absent* in Qwen2.5 at any size. The practical bridge is **French, the
> language of instruction**, not the L1. This is exactly why French is non-negotiable and
> why we do **not** ship a "local African language" UI: the model can render the buttons
> but cannot answer a document question coherently in those languages.

### The 4 chosen UI languages (+ English base)

| # | Language | Code | Script/dir | Why |
|---|----------|------|-----------|-----|
| 1 | **French** | `fr` | Latin / LTR | The origin scenario's language of schooling: Senegal CEB curriculum, and across francophone West/Central Africa + the Maghreb + Haiti — all low-connectivity. French is a top-tier Qwen language (in the IFEval-8) and near-zero UI cost (Latin). Non-negotiable. |
| 2 | **Arabic** | `ar` | Arabic / **RTL** | Unlocks MENA (Yemen, Sudan, rural Egypt) *and* the Sahel/Qur'anic-school layer sitting directly over the origin region. Among Qwen's better non-Latin languages: it is in the IFEval-8 and has its own AMMLU eval, unlike the African languages. The RTL cost is real but bounded (see §5) and demonstrating RTL is itself valuable. |
| 3 | **Spanish** | `es` | Latin / LTR | Opens all of rural Latin America (Andes, Central America, rural Mexico) — a huge offline audience. One of Qwen's strongest languages (IFEval-8), Latin script → near-zero UI cost. |
| 4 | **Portuguese** | `pt` | Latin / LTR | Lusophone Africa — Angola, Mozambique, and **Guinea-Bissau, which borders Senegal** — plus rural North/Northeast Brazil. IFEval-8 language, strong quality, Latin script. Chosen BR-neutral spelling where the two norms differ. |

Combined, these four + English cover the francophone origin region, the Arabic-speaking
Sahel/MENA, Latin America, and Lusophone Africa/Brazil — the four biggest offline /
low-connectivity blocs the model can actually serve well.

### Strong candidates rejected (and why)

- **Swahili, Hausa, Wolof, Bambara, Fula, Yoruba, Amharic** — REJECTED on quality, not
  demand. Even Qwen2.5-7B hits ~0% language consistency on the African languages it
  *does* list (Swahili/Amharic); Wolof/Bambara/Fula aren't in the model at all. At 1.5B
  a grounded answer in these languages is not achievable. Shipping them would be a broken
  promise. (This is the counter-intuitive result: the target region's own languages are
  the *least* viable.)
- **Hindi / Bengali / Urdu** — supported and high-demand (rural South Asia), but not in
  the IFEval-8 trusted set, and Devanagari/Bengali/Arabic-script generation at 1.5B is
  markedly weaker than the Latin high-resource tier. **Hindi is the strongest
  "reconsider for phase 2"** candidate if South Asia becomes a target — but it should be
  quality-tested on real documents first.
- **Indonesian** — the closest near-miss: Latin script (near-zero UI cost), squarely in
  the IFEval-8, and a genuinely offline archipelago audience (~200M). Rejected only for
  slot count: Spanish+Portuguese+Arabic give broader combined population and align with
  the Africa/Sahel origin. **Top runner-up for a 5th language.**
- **Vietnamese** — in the IFEval-8, Latin script, real offline SE-Asia reach; edged out
  by the same slot-count logic as Indonesian.
- **Chinese** — Qwen's single strongest language, but not an offline/low-connectivity
  target audience for this deployment. Rejected on deployment-fit, not quality.
- **German / Italian / Russian / Dutch / Polish / etc.** — strong Qwen quality but
  wealthy, high-connectivity regions: off-mission for a device built for the
  disconnected.

---

## 2. Backend note: the system prompt is already multilingual-correct

`app/server/cag.py` → `SYSTEM_TMPL` currently ends with:

> "…Reply in the **same language as the question**."

**This is correct for a multilingual UI and needs no change.** The reply language is tied
to the *question's* language, fully independent of the UI chrome language — a child can
run a Spanish UI over a French document and still get French answers if they ask in
French. No backend edit required.

- Optional, non-blocking nicety: rewording to "the language of the user's question" is
  functionally identical; skip it unless touching that file for another reason.
- **Do NOT** change it to "language of the UI" — that would break the common case of a
  document in one language read through a UI in another.

The guided-preset instruction is different by design. In `index.html` the `LANG` const is
`"Reply in the same language as the document."` — because a one-tap "Summarize" has no
typed question to infer a language from, so it anchors to the *document's* language. Keep
that behavior: when presets are translated (§4), the preset *body* is written in the UI
language for naturalness, but it must still end with the **document-language** instruction
(the localized `LANG` string), so "Summarize" over a French fiche still answers in French.

---

## 3. Implementation architecture (single-file, no framework)

Match the house style: plain DOM, `getElementById`, no build step.

### 3.1 The `T` table and `t()` accessor

```js
const T = {
  en: { /* key: "English string", ... */ },
  fr: { /* … */ }, ar: { /* … */ }, es: { /* … */ }, pt: { /* … */ },
};
const LANGS = [
  { code: "en", label: "English",   rtl: false },
  { code: "fr", label: "Français",  rtl: false },
  { code: "ar", label: "العربية",   rtl: true  },
  { code: "es", label: "Español",   rtl: false },
  { code: "pt", label: "Português", rtl: false },
];
let LANG = pickInitialLang();
const t = (k, ...a) => {
  const v = (T[LANG] && T[LANG][k]) ?? T.en[k] ?? k;   // fall back en → key
  return typeof v === "function" ? v(...a) : v;         // templated strings are fns
};
function pickInitialLang() {
  const saved = localStorage.getItem("cag_lang");
  if (saved && T[saved]) return saved;
  const nav = (navigator.language || "en").slice(0, 2).toLowerCase();
  return T[nav] ? nav : "fr";   // default FR (origin scenario), not EN
}
```

Interpolated strings (character counts, error templates with numbers) are stored as
**functions**, e.g. `over_cap: (tok, cap) => \`Le document fait ${tok}…\``.

### 3.2 Static markup: `data-i18n` sweep

Tag every static user-facing element in the HTML with a data-attribute and run one sweep
in `applyLang()`:

- `data-i18n="key"` → sets `textContent`
- `data-i18n-ph="key"` → sets `placeholder`
- `data-i18n-title="key"` → sets `title`
- `data-i18n-aria="key"` → sets `aria-label`

```js
function applyLang(code) {
  LANG = code;
  localStorage.setItem("cag_lang", code);
  const rtl = LANGS.find(l => l.code === code)?.rtl;
  document.documentElement.lang = code;
  document.documentElement.dir = rtl ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach(
    el => el.textContent = t(el.dataset.i18n));
  document.querySelectorAll("[data-i18n-ph]").forEach(
    el => el.placeholder = t(el.dataset.i18nPh));
  document.querySelectorAll("[data-i18n-title]").forEach(
    el => el.title = t(el.dataset.i18nTitle));
  document.querySelectorAll("[data-i18n-aria]").forEach(
    el => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  updateIngestMeter();   // recompute "N characters"/"~N tokens" strings
  renderPresets();       // rebuild chips in new language
  renderEmptyIfNeeded(); // rebuild suggestion cards
  // statusbar labels are static (data-i18n); dynamic values re-render on next /status
}
```

### 3.3 Dynamic nodes use `t()` directly

Anything built in JS (suggestion cards, preset chips, library rows, `alert`/`confirm`/
`prompt`, streamed error text, copy-button label, empty state) must call `t("key")`
instead of hardcoded English. The current hardcoded literals in `SUGGESTIONS`, `PRESETS`,
`renderLibList`, `switchDoc`, `deleteDoc`, `showLibrary`, `ask`, `showDocs` all move into
`T`.

### 3.4 Language picker

- **Placement:** a compact `<select id="langSel">` in the header, styled like the existing
  `.model-pill` (add a `.lang-pill` rule reusing the pill look), inserted right after the
  brand / before the model tag. Also mirror it once on the ingest view header area so a
  first-time user can switch before ingesting.
- Options are the endonyms from `LANGS` (English / Français / العربية / Español /
  Português). On `change` → `applyLang(select.value)`.
- **Persistence:** `localStorage["cag_lang"]`. Initial value from §3.1
  (`saved → navigator.language → "fr"`).
- Set the picker's value from `LANG` on boot, and call `applyLang(LANG)` once at the end
  of the boot IIFE (after `renderPresets()`), so first paint is already localized.

### 3.5 Error-code → localized message map

The backend returns a stable `error` **code** plus structured fields; the front rebuilds a
localized message from the code and ignores the server's English `message` when the code
is known (falling back to it, then to `HTTP <status>`, when unknown).

```js
function localizeError(j, httpStatus) {
  const code = j && j.error;
  const E = (T[LANG] && T[LANG].err) || T.en.err;
  switch (code) {
    case "over_cap":     return E.over_cap(j.tokens, j.cap);
    case "library_full": return E.library_full(j.count, j.cap);
    case "not_ingested": return E.not_ingested;
    case "unknown_doc":  return E.unknown_doc;
    case "no_doc":       return E.no_doc;
    case "bad_pack":     return E.bad_pack;
    case "llama_unreachable": return E.llama_unreachable;
    case "empty":        return E.empty;          // "no text / no question"
    default: return (j && j.message) || t("http_status", httpStatus);
  }
}
```

**Enumerated backend codes** (from `app/server/main.py`) and the fields each carries:

| Code | Where raised | Structured fields available | Localized as |
|------|--------------|-----------------------------|--------------|
| `empty` | ingest / ask / switch (missing text, question, or id) | — | generic "nothing to send" (`err.empty`) |
| `library_full` | `/api/ingest` (≥ `MAX_DOCS`=20) | `count`, `cap` | `err.library_full(count, cap)` |
| `over_cap` | `/api/ingest` (> `FICHE_CAP`=12000 tok) | `tokens`, `cap` | `err.over_cap(tokens, cap)` |
| `llama_unreachable` | ingest / switch (server down / prefill fail) | `message` (detail) | `err.llama_unreachable` (generic; drop server detail) |
| `not_ingested` | `/api/ask` before any ingest | — | `err.not_ingested` |
| `no_doc` | `/api/doc` (CONTAINER.md missing) | — | `err.no_doc` |
| `unknown_doc` | `/api/switch`, `DELETE /api/library/{id}` | — | `err.unknown_doc` |
| `bad_pack` | `/api/pack/import` (malformed tar) | `message` (detail) | `err.bad_pack` (generic) |

Front-only errors (network failures, `AbortError`) never reach the backend and use the
plain UI keys (`neterr`, `stopped`, etc.) in the table below.

---

## 4. Complete translation table (~40 UI strings + error codes)

Notes on conventions:
- Product/brand name **"Cache-Augmented Generation"** stays untranslated in all languages.
- Templated entries show the JS function; `{n}`/`{tok}` mark interpolation.
- **The literal token `DELETE`** in the wipe prompt stays untranslated in every language
  (the code compares `typed !== "DELETE"`); only the surrounding sentence is localized.
- Portuguese uses BR-neutral vocabulary; where PT-PT/PT-BR diverge sharply the more
  widely-understood form is chosen.

### Header / navigation

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `nav.library` | Library | Bibliothèque | المكتبة | Biblioteca | Biblioteca |
| `nav.library.title` | Your document library | Votre bibliothèque de documents | مكتبة مستنداتك | Tu biblioteca de documentos | A sua biblioteca de documentos |
| `nav.docs` | Docs | Aide | الدليل | Ayuda | Ajuda |
| `nav.docs.title` | How this app works | Comment fonctionne l'application | كيف يعمل هذا التطبيق | Cómo funciona la aplicación | Como funciona a aplicação |
| `nav.newdoc` | New document | Nouveau document | مستند جديد | Nuevo documento | Novo documento |
| `nav.newdoc.title` | Load a different document | Charger un autre document | تحميل مستند آخر | Cargar otro documento | Carregar outro documento |
| `nav.clear` | Clear chat | Effacer la discussion | مسح المحادثة | Borrar el chat | Limpar a conversa |
| `nav.clear.title` | Clear the conversation | Effacer la conversation | مسح المحادثة | Borrar la conversación | Limpar a conversa |

### Ingest view

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `ingest.eyebrow` | Step 1 · Ingest | Étape 1 · Charger | الخطوة 1 · التحميل | Paso 1 · Cargar | Passo 1 · Carregar |
| `ingest.h2` | Load a document to chat about | Chargez un document pour en discuter | حمّل مستندًا لتتحدث عنه | Carga un documento para conversar sobre él | Carregue um documento para conversar sobre ele |
| `ingest.lede` | Paste the text you want to ask questions about. It's prefilled into the model's cache once and reused for every question — so answers stay fast and grounded in your document. | Collez le texte sur lequel vous voulez poser des questions. Il est chargé une seule fois dans la mémoire du modèle, puis réutilisé à chaque question — les réponses restent rapides et fidèles à votre document. | الصق النص الذي تريد طرح الأسئلة حوله. يُحمَّل مرة واحدة في ذاكرة النموذج ثم يُعاد استخدامه مع كل سؤال، لتبقى الإجابات سريعة ومستندة إلى وثيقتك. | Pega el texto sobre el que quieres hacer preguntas. Se carga una sola vez en la memoria del modelo y se reutiliza en cada pregunta, así las respuestas son rápidas y fieles a tu documento. | Cole o texto sobre o qual quer fazer perguntas. Ele é carregado uma única vez na memória do modelo e reutilizado em cada pergunta — assim as respostas continuam rápidas e fiéis ao seu documento. |
| `ingest.title.ph` | Optional title (e.g. Le cycle de l'eau) | Titre (facultatif, ex. : Le cycle de l'eau) | عنوان (اختياري، مثل: دورة الماء) | Título (opcional, p. ej.: El ciclo del agua) | Título (opcional, ex.: O ciclo da água) |
| `ingest.fiche.ph` | Paste your document here… | Collez votre document ici… | الصق مستندك هنا… | Pega tu documento aquí… | Cole o seu documento aqui… |
| `ingest.chars` *(fn n)* | `{n} character` / `{n} characters` | `{n} caractère` / `{n} caractères` | `{n} حرف` | `{n} carácter` / `{n} caracteres` | `{n} caractere` / `{n} caracteres` |
| `ingest.meter.title` | Estimated fill of the ~12,000 token cache | Remplissage estimé du cache (~12 000 jetons) | الامتلاء التقديري للذاكرة (~12000 رمز) | Llenado estimado de la caché (~12 000 tokens) | Preenchimento estimado da cache (~12 000 tokens) |
| `ingest.tokens` *(fn n)* | ~{n} tokens | ~{n} jetons | ~{n} رمز | ~{n} tokens | ~{n} tokens |
| `ingest.btn` | Ingest document | Charger le document | تحميل المستند | Cargar documento | Carregar documento |
| `ingest.status.prefill` | Prefilling cache… | Préparation du cache… | جارٍ تحضير الذاكرة… | Preparando la caché… | A preparar a cache… |
| `ingest.err.empty` | Paste some text first. | Collez d'abord du texte. | الصق نصًا أولًا. | Primero pega un texto. | Cole primeiro um texto. |
| `ingest.err.failed` | Ingest failed. | Échec du chargement. | فشل التحميل. | Error al cargar. | Falha ao carregar. |

### Chat status bar

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `status.document` | document | document | المستند | documento | documento |
| `status.history` | history | historique | السجل | historial | histórico |
| `status.tok` | tok | jetons | رمز | tok | tok |
| `status.restored` | cache restored from disk | cache restauré depuis le disque | تمت استعادة الذاكرة من القرص | caché restaurada desde el disco | cache restaurada do disco |

### Composer

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `composer.ph` | Ask about the document… | Posez une question sur le document… | اطرح سؤالًا عن المستند… | Pregunta sobre el documento… | Faça uma pergunta sobre o documento… |
| `composer.send.title` | Send (Enter) | Envoyer (Entrée) | إرسال (Enter) | Enviar (Intro) | Enviar (Enter) |
| `composer.send.aria` | Send | Envoyer | إرسال | Enviar | Enviar |
| `composer.stop.title` | Stop generating | Arrêter la génération | إيقاف التوليد | Detener la generación | Parar a geração |
| `composer.stop.aria` | Stop | Arrêter | إيقاف | Detener | Parar |
| `composer.hint` *(HTML w/ kbd)* | `<kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line` | `<kbd>Entrée</kbd> pour envoyer · <kbd>Maj</kbd>+<kbd>Entrée</kbd> pour un saut de ligne` | `<kbd>Enter</kbd> للإرسال · <kbd>Shift</kbd>+<kbd>Enter</kbd> لسطر جديد` | `<kbd>Intro</kbd> para enviar · <kbd>Mayús</kbd>+<kbd>Intro</kbd> para una nueva línea` | `<kbd>Enter</kbd> para enviar · <kbd>Shift</kbd>+<kbd>Enter</kbd> para nova linha` |
| `footer` | Runs fully on-device · llama.cpp KV-cache slot reuse | Fonctionne entièrement sur l'appareil · réutilisation du cache KV llama.cpp | يعمل بالكامل على الجهاز · إعادة استخدام ذاكرة KV في llama.cpp | Funciona totalmente en el dispositivo · reutilización de la caché KV de llama.cpp | Funciona totalmente no dispositivo · reutilização da cache KV do llama.cpp |

*(`composer.hint` contains `<kbd>` markup, so set it via `innerHTML`, not `textContent` —
give it a distinct handling in `applyLang` or use a dedicated `data-i18n-html` attribute.)*

### Empty state + suggestion cards

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `empty.h3` | Ask anything about your document | Posez n'importe quelle question sur votre document | اسأل أي شيء عن مستندك | Pregunta lo que quieras sobre tu documento | Pergunte o que quiser sobre o seu documento |
| `empty.p` | Answers are grounded in the text you ingested. Try one of these to start: | Les réponses s'appuient uniquement sur le texte chargé. Essayez l'une de ces suggestions pour commencer : | تعتمد الإجابات على النص الذي حمّلته. جرّب أحد هذه الاقتراحات للبدء: | Las respuestas se basan en el texto que cargaste. Prueba una de estas para empezar: | As respostas baseiam-se no texto que carregou. Experimente uma destas para começar: |
| `sug1.t` | Summarize this document | Résume ce document | لخّص هذا المستند | Resume este documento | Resume este documento |
| `sug1.h` | Key points in a few bullets | Les points clés en quelques puces | أهم النقاط في بضع نقاط | Los puntos clave en unas viñetas | Os pontos-chave em alguns tópicos |
| `sug2.t` | What are the main takeaways? | Quelles sont les idées principales ? | ما هي الأفكار الرئيسية؟ | ¿Cuáles son las ideas principales? | Quais são as ideias principais? |
| `sug2.h` | The essentials, briefly | L'essentiel, en bref | الأساسيات باختصار | Lo esencial, en breve | O essencial, em resumo |
| `sug3.t` | Draft 3 questions this doc answers | Propose 3 questions auxquelles ce document répond | اقترح 3 أسئلة يجيب عنها هذا المستند | Propón 3 preguntas que este documento responde | Sugere 3 perguntas que este documento responde |
| `sug3.h` | Explore what's inside | Explorez son contenu | استكشف محتواه | Explora su contenido | Explore o conteúdo |
| `sug4.t` | Explain the hardest part simply | Explique simplement la partie la plus difficile | اشرح الجزء الأصعب ببساطة | Explica de forma sencilla la parte más difícil | Explica de forma simples a parte mais difícil |
| `sug4.h` | Plain-language walkthrough | Une explication en mots simples | شرح بلغة مبسّطة | Una explicación con palabras simples | Uma explicação em palavras simples |

### Guided-mode preset chips (label + prompt body)

The `label` is what the chip shows; the `body` is the prompt fired at the model, and must
be concatenated with the localized `LANG` (document-language) instruction — see §2.

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `preset.quiz.label` | Quiz me | Interroge-moi | اختبرني | Ponme a prueba | Faz-me perguntas |
| `preset.quiz.body` | Ask me 3 questions about the document, one at a time; wait for my answer before the next one and tell me if I am right, correcting me from the document. | Pose-moi 3 questions sur le document, une par une ; attends ma réponse avant la suivante et dis-moi si j'ai raison, en me corrigeant à partir du document. | اطرح عليّ 3 أسئلة عن المستند، سؤالًا واحدًا في كل مرة؛ وانتظر إجابتي قبل السؤال التالي، وأخبرني إن كنت مصيبًا، مصحّحًا لي من المستند. | Hazme 3 preguntas sobre el documento, una por una; espera mi respuesta antes de la siguiente y dime si acierto, corrigiéndome con el documento. | Faz-me 3 perguntas sobre o documento, uma de cada vez; espera pela minha resposta antes da seguinte e diz-me se acertei, corrigindo-me com base no documento. |
| `preset.simpler.label` | Explain simpler | Explique plus simplement | اشرح ببساطة | Explícalo más fácil | Explica mais simples |
| `preset.simpler.body` | Explain the document in simpler terms, as if to a beginner, with a short everyday example. | Explique le document en termes plus simples, comme à un débutant, avec un court exemple de la vie quotidienne. | اشرح المستند بعبارات أبسط، كأنك تشرح لمبتدئ، مع مثال قصير من الحياة اليومية. | Explica el documento en términos más sencillos, como a un principiante, con un breve ejemplo cotidiano. | Explica o documento em termos mais simples, como a um iniciante, com um breve exemplo do dia a dia. |
| `preset.exercises.label` | Make 5 exercises | Crée 5 exercices | أنشئ 5 تمارين | Crea 5 ejercicios | Cria 5 exercícios |
| `preset.exercises.body` | Create 5 exercises based on the document, from easy to hard, then give an answer key at the end. | Crée 5 exercices à partir du document, du plus facile au plus difficile, puis donne le corrigé à la fin. | أنشئ 5 تمارين مستندة إلى المستند، من الأسهل إلى الأصعب، ثم قدّم الإجابات النموذجية في النهاية. | Crea 5 ejercicios basados en el documento, de fácil a difícil, y da las soluciones al final. | Cria 5 exercícios com base no documento, do fácil ao difícil, e dá a chave de respostas no final. |
| `preset.summary.label` | Summarize | Résume | لخّص | Resume | Resume |
| `preset.summary.body` | Summarize the document in a few clear bullet points. | Résume le document en quelques points clairs. | لخّص المستند في بضع نقاط واضحة. | Resume el documento en unos pocos puntos claros. | Resume o documento em alguns tópicos claros. |
| `preset.lang` *(appended)* | Reply in the same language as the document. | Réponds dans la même langue que le document. | أجب بنفس لغة المستند. | Responde en el mismo idioma que el documento. | Responde no mesmo idioma do documento. |

### Message actions / streaming states

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `msg.copy` | Copy | Copier | نسخ | Copiar | Copiar |
| `msg.copied` | Copied | Copié | تم النسخ | Copiado | Copiado |
| `msg.noanswer` | (no answer) | (aucune réponse) | (لا توجد إجابة) | (sin respuesta) | (sem resposta) |
| `msg.stopped` | Stopped. | Arrêté. | تم الإيقاف. | Detenido. | Parado. |
| `neterr` *(prefix)* | Network error:  | Erreur réseau :  | خطأ في الشبكة:  | Error de red:  | Erro de rede:  |
| `switch.restoring` | Restoring cache… | Restauration du cache… | جارٍ استعادة الذاكرة… | Restaurando la caché… | A restaurar a cache… |

### Docs modal

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `docs.head` | How this app works | Comment fonctionne l'application | كيف يعمل هذا التطبيق | Cómo funciona la aplicación | Como funciona a aplicação |
| `modal.close` | Close | Fermer | إغلاق | Cerrar | Fechar |
| `docs.unavail` | Docs unavailable. | Aide indisponible. | الدليل غير متاح. | Ayuda no disponible. | Ajuda indisponível. |

### Library modal

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `lib.head` | Document library | Bibliothèque de documents | مكتبة المستندات | Biblioteca de documentos | Biblioteca de documentos |
| `lib.empty` | No documents yet. Ingest one, or import a content pack. | Aucun document pour l'instant. Chargez-en un ou importez un pack de contenu. | لا توجد مستندات بعد. حمّل واحدًا أو استورد حزمة محتوى. | Aún no hay documentos. Carga uno o importa un paquete de contenido. | Ainda não há documentos. Carregue um ou importe um pacote de conteúdo. |
| `lib.untitled` | (untitled) | (sans titre) | (بدون عنوان) | (sin título) | (sem título) |
| `lib.active` | active | actif | نشط | activo | ativo |
| `lib.meta` *(fn tok, active)* | `{tok} tok` [`· active`] | `{tok} jetons` [`· actif`] | `{tok} رمز` [`· نشط`] | `{tok} tok` [`· activo`] | `{tok} tok` [`· ativo`] |
| `lib.switch` | Switch | Ouvrir | فتح | Abrir | Abrir |
| `lib.delete` | Delete | Supprimer | حذف | Eliminar | Eliminar |
| `lib.export` | Export pack | Exporter un pack | تصدير حزمة | Exportar paquete | Exportar pacote |
| `lib.export.title` | Download the whole library as a .tar pack | Télécharger toute la bibliothèque en pack .tar | تنزيل المكتبة كاملة كحزمة ‎.tar | Descargar toda la biblioteca como paquete .tar | Descarregar toda a biblioteca como pacote .tar |
| `lib.import` | Import pack | Importer un pack | استيراد حزمة | Importar paquete | Importar pacote |
| `lib.import.title` | Import a .tar content pack | Importer un pack de contenu .tar | استيراد حزمة محتوى ‎.tar | Importar un paquete de contenido .tar | Importar um pacote de conteúdo .tar |
| `lib.importing` | Importing… | Importation… | جارٍ الاستيراد… | Importando… | A importar… |
| `lib.wipe` | Wipe everything | Tout effacer | مسح كل شيء | Borrar todo | Apagar tudo |
| `lib.wipe.title` | Delete everything on the device | Supprimer tout ce qui se trouve sur l'appareil | حذف كل شيء على الجهاز | Eliminar todo lo que hay en el dispositivo | Eliminar tudo o que está no dispositivo |

### Confirm / prompt dialogs + action errors

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `del.confirm` *(fn title)* | Delete "{title}"? Its cache and chat are removed from the device. | Supprimer « {title} » ? Son cache et sa discussion seront retirés de l'appareil. | حذف «{title}»؟ ستُزال ذاكرته ومحادثته من الجهاز. | ¿Eliminar «{title}»? Su caché y su chat se quitarán del dispositivo. | Eliminar «{title}»? A sua cache e conversa serão removidas do dispositivo. |
| `del.failed` | Delete failed. | Échec de la suppression. | فشل الحذف. | Error al eliminar. | Falha ao eliminar. |
| `del.error` *(prefix)* | Delete error:  | Erreur de suppression :  | خطأ في الحذف:  | Error al eliminar:  | Erro ao eliminar:  |
| `switch.failed` | Switch failed. | Échec de l'ouverture. | فشل الفتح. | Error al abrir. | Falha ao abrir. |
| `switch.error` *(prefix)* | Switch error:  | Erreur d'ouverture :  | خطأ في الفتح:  | Error al abrir:  | Erro ao abrir:  |
| `import.failed` | Import failed. | Échec de l'importation. | فشل الاستيراد. | Error al importar. | Falha ao importar. |
| `import.error` *(prefix)* | Import error:  | Erreur d'importation :  | خطأ في الاستيراد:  | Error al importar:  | Erro ao importar:  |
| `wipe.prompt` | This permanently deletes ALL documents, caches and chats on this device.\nType DELETE to confirm. | Cette action supprime définitivement TOUS les documents, caches et discussions de cet appareil.\nTapez DELETE pour confirmer. | سيؤدي هذا إلى حذف جميع المستندات والذاكرة والمحادثات نهائيًا من هذا الجهاز.\nاكتب DELETE للتأكيد. | Esto elimina de forma permanente TODOS los documentos, cachés y chats de este dispositivo.\nEscribe DELETE para confirmar. | Isto elimina permanentemente TODOS os documentos, caches e conversas deste dispositivo.\nEscreva DELETE para confirmar. |
| `wipe.error` *(prefix)* | Wipe error:  | Erreur d'effacement :  | خطأ في المسح:  | Error al borrar:  | Erro ao apagar:  |

### Backend error-code messages (`T[lang].err.*`)

| key | en | fr | ar | es | pt |
|-----|----|----|----|----|----|
| `err.empty` | Nothing to send. | Rien à envoyer. | لا يوجد ما يُرسَل. | No hay nada que enviar. | Nada para enviar. |
| `err.not_ingested` | Ingest a document before asking questions. | Chargez un document avant de poser des questions. | حمّل مستندًا قبل طرح الأسئلة. | Carga un documento antes de hacer preguntas. | Carregue um documento antes de fazer perguntas. |
| `err.unknown_doc` | No such document in the library. | Ce document n'existe pas dans la bibliothèque. | لا يوجد هذا المستند في المكتبة. | No existe ese documento en la biblioteca. | Não existe esse documento na biblioteca. |
| `err.no_doc` | The help document isn't available in this build. | Le document d'aide n'est pas disponible dans cette version. | وثيقة المساعدة غير متاحة في هذه النسخة. | El documento de ayuda no está disponible en esta versión. | O documento de ajuda não está disponível nesta versão. |
| `err.bad_pack` | This file isn't a valid content pack. | Ce fichier n'est pas un pack de contenu valide. | هذا الملف ليس حزمة محتوى صالحة. | Este archivo no es un paquete de contenido válido. | Este ficheiro não é um pacote de conteúdo válido. |
| `err.llama_unreachable` | The AI engine isn't responding. Please try again in a moment. | Le moteur d'IA ne répond pas. Réessayez dans un instant. | محرّك الذكاء الاصطناعي لا يستجيب. حاول مرة أخرى بعد قليل. | El motor de IA no responde. Inténtalo de nuevo en un momento. | O motor de IA não está a responder. Tente novamente dentro de momentos. |
| `err.library_full` *(fn count, cap)* | The library already holds {count} documents (max {cap}). Delete one before ingesting another — each saved cache is ~300 MB on disk. | La bibliothèque contient déjà {count} documents (max {cap}). Supprimez-en un avant d'en charger un autre — chaque cache enregistré occupe ~300 Mo. | تحتوي المكتبة بالفعل على {count} مستندات (الحد الأقصى {cap}). احذف واحدًا قبل تحميل آخر — كل ذاكرة محفوظة تشغل نحو 300 ميغابايت. | La biblioteca ya tiene {count} documentos (máx. {cap}). Elimina uno antes de cargar otro: cada caché guardada ocupa ~300 MB. | A biblioteca já tem {count} documentos (máx. {cap}). Elimine um antes de carregar outro — cada cache guardada ocupa ~300 MB. |
| `err.over_cap` *(fn tokens, cap)* | Document is {tokens} tokens, over the {cap}-token cap. The whole document is held in the model's cache, which must fit the device's RAM and still leave room for the chat. Shorten the text and try again. | Le document fait {tokens} jetons, au-delà de la limite de {cap}. Tout le document est gardé dans le cache du modèle, qui doit tenir dans la mémoire de l'appareil tout en laissant de la place pour la discussion. Raccourcissez le texte et réessayez. | المستند يحتوي على {tokens} رمزًا، وهو يتجاوز الحد الأقصى {cap}. تُحفظ الوثيقة كاملة في ذاكرة النموذج، التي يجب أن تتّسع في ذاكرة الجهاز مع ترك مساحة للمحادثة. اختصر النص وحاول مجددًا. | El documento tiene {tokens} tokens, por encima del límite de {cap}. Todo el documento se guarda en la caché del modelo, que debe caber en la memoria del dispositivo y dejar espacio para el chat. Acorta el texto e inténtalo de nuevo. | O documento tem {tokens} tokens, acima do limite de {cap}. Todo o documento fica na cache do modelo, que precisa de caber na memória do dispositivo e ainda deixar espaço para a conversa. Encurte o texto e tente novamente. |
| `http_status` *(fn code)* | HTTP {code} | HTTP {code} | HTTP {code} | HTTP {code} | HTTP {code} |

---

## 5. RTL handling for Arabic

The single-file page is almost entirely flexbox + text, so the bulk of mirroring comes
free from one attribute flip. `applyLang` already sets:

```js
document.documentElement.dir = rtl ? "rtl" : "ltr";
document.documentElement.lang = code;
```

**What flips automatically under `dir="rtl"`** (no work): header row order, text
alignment, the chat message rows (assistant→right / user→left via flex `start`/`end`),
the composer (send button moves to the left of the textarea), status bar, modal layouts,
suggestion grid.

**The handful of *physical* CSS properties to convert to logical** (so RTL is clean, not
just mirrored-flex) — these are the only real cost:

| Current (physical) | Change to (logical) | Where |
|--------------------|---------------------|-------|
| `padding-left: 22px` on `.body ul, .body ol` | `padding-inline-start: 22px` | markdown lists |
| `border-left: 3px …; padding: … 14px` on `.body blockquote` | `border-inline-start` + `padding-inline-start` | markdown quotes |
| `.msg.user .body { border-radius: 14px 14px 4px 14px }` | add an `:dir(rtl)` / `[dir=rtl]` override mirroring the corner (`14px 14px 14px 4px`) | user bubble tail |
| `.composer-hint kbd`, `.s-hint` etc. | already symmetric — no change | — |

Add one scoped block instead of touching each rule:

```css
[dir="rtl"] .body ul, [dir="rtl"] .body ol { padding-inline-start: 22px; padding-left: 0; }
[dir="rtl"] .msg.user .body { border-radius: 14px 14px 14px 4px; }
```

**Bidi hygiene:** Latin runs embedded in Arabic (the model-name pill, `~N رمز`, the
`.tar` extension, `llama.cpp`) should not scramble the line. The `.tar`/`Enter` strings in
the table already carry a leading `‎` (LRM) where needed; for the model pill add
`unicode-bidi: plaintext` to `.model-pill` so it isolates. The progress meters
(`#tokenFill`, `#histfill`, `.meter`) fill left-to-right; under RTL they will visually fill
from the right, which is the *correct* RTL reading direction — leave them.

**Digits:** keep Western Arabic numerals (0-9) in Arabic UI — they are standard in
Maghreb/Sahel usage and `toLocaleString()` without a locale already yields them. Do not
force Eastern Arabic-Indic digits.

Net RTL cost: ~1 attribute toggle + ~4 lines of scoped CSS + one `unicode-bidi`. Small and
contained — which is why Arabic clears the bar despite being the only RTL language.

---

## 6. Implementation checklist (for the implementer)

1. Add `T`, `LANGS`, `t()`, `pickInitialLang()`, `applyLang()` near the top of the
   `<script>`.
2. Sweep the static markup: add `data-i18n` / `-ph` / `-title` / `-aria` / `-html`
   attributes to every user-facing element listed in §4 (header buttons, ingest view,
   status bar labels, composer, footer).
3. Replace hardcoded literals in `SUGGESTIONS`, `PRESETS` (+ `LANG` const), `renderLibList`,
   `showLibrary`, `deleteDoc`, `switchDoc`, `showDocs`, `ask`, `updateIngestMeter`,
   `renderStatus` (`" tok"`, restored chip) with `t(...)` calls.
4. Add the `<select id="langSel">` picker to the header (and a mirror on ingest); wire
   `change → applyLang`; add `.lang-pill` styling reusing `.model-pill`.
5. Add `localizeError()` (§3.5) and route both error paths (`ask` non-OK + SSE `j.error`,
   and every `alert(s.message …)`) through it.
6. Add the scoped `[dir="rtl"]` CSS block (§5) + `unicode-bidi: plaintext` on
   `.model-pill`.
7. Call `applyLang(LANG)` once at the end of the boot IIFE (after `renderPresets()`), and
   set `langSel.value = LANG`.
8. Backend: **no change required** (§2). Leave `SYSTEM_TMPL` as-is.

No backend, Docker, or model changes. Fully offline-compatible: all strings ship inside
the one static HTML file — no fonts or assets fetched (Arabic renders with the system font
stack; the OS on the Pi/host provides Arabic glyphs).
