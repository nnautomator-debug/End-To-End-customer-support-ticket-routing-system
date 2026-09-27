# Column Analysis — Customer Support Ticket Priority Classification

**Dataset:** `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
**Rows:** 28,587  |  **Columns:** 16
**Target variable:** `priority` (`low` / `medium` / `high`)
**Rule followed in this document:** purely descriptive — no column changed, no transformation, no feature engineering.

---

## 1. Dataset at a glance


| Column     | Dtype  | Role                    | Nulls  | Null % | Unique | Mean / Median (where numeric)            |
| ---------- | ------ | ----------------------- | ------ | ------ | ------ | ---------------------------------------- |
| `subject`  | object | Text — ticket subject   | 3,838  | 13.43% | 24,749 | char: 44.5 / 42.0 · words: 5.4 / 5.0     |
| `body`     | object | Text — ticket message   | 0      | 0.00%  | 28,587 | char: 387.3 / 386.0 · words: 53.5 / 54.0 |
| `answer`   | object | Text — support reply    | 7      | 0.02%  | 28,580 | char: 387.3 / 390.0 · words: 56.6 / 58.0 |
| `type`     | object | Ticket category (4)     | 0      | 0.00%  | 4      | mode: `Incident` (40.11%)                |
| `queue`    | object | Routing queue (10)      | 0      | 0.00%  | 10     | mode: `Technical Support` (29.25%)       |
| `priority` | object | **TARGET** (3)          | 0      | 0.00%  | 3      | mode: `medium` (40.28%)                  |
| `language` | object | Language of ticket (2)  | 0      | 0.00%  | 2      | mode: `en` (57.15%)                      |
| `version`  | int64  | Version id (3 discrete) | 0      | 0.00%  | 3      | **278.4 / 400.0**                        |
| `tag_1`    | object | Primary topic tag       | 0      | 0.00%  | 116    | mode: `Security` (5,880)                 |
| `tag_2`    | object | Secondary topic tag     | 13     | 0.05%  | 256    | mode: `Disruption` (1,188)               |
| `tag_3`    | object | Topic tag               | 136    | 0.48%  | 392    | –                                        |
| `tag_4`    | object | Topic tag               | 3,058  | 10.70% | 554    | –                                        |
| `tag_5`    | object | Topic tag               | 14,042 | 49.12% | 602    | –                                        |
| `tag_6`    | object | Topic tag               | 22,713 | 79.45% | 575    | –                                        |
| `tag_7`    | object | Topic tag               | 26,547 | 92.86% | 427    | –                                        |
| `tag_8`    | object | Topic tag               | 28,022 | 98.02% | 224    | –                                        |


- **Fully duplicated rows:** 0
- **Duplicate** `subject` **pairs:** 3,837 (all of them are the *missing* subjects, `NaN == NaN` in pandas — the same rows) → **no genuine duplicates**
- **Duplicate** `(subject + body)` **pairs:** 0



### Data-quality red flags (from the top)

1. `subject` is missing in **3,838 rows (13.43%)**.
2. `answer` is missing in 7 rows.
3. Tags are a *ragged multi-label*: `tag_5`→`tag_8` are mostly empty (49%→98% missing). Every row has at least 1 tag; the most common row carries **4 tags (10,984 rows)**.
4. Version is nearly constant — **65% of rows = 400**, three values total.

---



## 2. Target variable — `priority`


| Level  | Count  | % of total |
| ------ | ------ | ---------- |
| medium | 11,515 | 40.28%     |
| high   | 11,178 | 39.10%     |
| low    | 5,894  | 20.62%     |


- Reasonably balanced for a 3-class problem (high ≈ medium ≈ 40%, low ≈ 21%).
- Low is the minority class (~1:2 vs the others) → class weighting / stratified splits advisable later, but no class is negligible.

---



## 3. Column-by-column analysis



### 3.1 `subject` — ticket subject (text)


| Measure                     | Value                              |
| --------------------------- | ---------------------------------- |
| Non-null count              | 24,749                             |
| **Null count / %**          | **3,838 / 13.43%**                 |
| Unique                      | 24,749 (all present values unique) |
| Char length mean / median   | 44.52 / 42.0                       |
| Char length std / min / max | 19.52 / 3 / 675                    |
| Word count mean / median    | 5.38 / 5.0                         |
| Word count min / max        | 1 / 77                             |
| Empty strings               | 0 (only `NaN`)                     |
| Newline / tag chars         | none                               |


**Impact on target — evidence:** Practically **none of the length statistics differ by priority**.


| priority | n (non-null) | mean chars | median chars |
| -------- | ------------ | ---------- | ------------ |
| high     | 9,655        | 44.11      | 42.0         |
| medium   | 9,978        | 44.75      | 43.0         |
| low      | 5,116        | 44.87      | 42.0         |


The *semantic content* of the subject is clearly predictive in the sample rows (e.g. "Wesentlicher Sicherheitsvorfall" → high), but subject **length is not**. Frequency check: the 3,838 missing subjects are unique-looking — no dominant repeated template beyond `NaN`.

**Correction strategies:** impute missing `subject` from the first sentence of `body`, or treat it as a separate pattern for the model to learn; subjects are near-unique so the raw string is high-cardinality → for modelling it should be embedded rather than one-hot encoded.

### 3.2 `body` — ticket body (text, multilingual)


| Measure                              | Value                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| Non-null count                       | 28,587                                                                               |
| **Null count / %**                   | **0 / 0.00%**                                                                        |
| Unique                               | 28,587 (perfectly unique — no template duplicates)                                   |
| Char length mean / median            | 387.26 / 386.0                                                                       |
| Char length std / min / max          | 200.13 / 6 / 1,469                                                                   |
| Word count mean / median             | 53.5 / 54.0                                                                          |
| Word count min / max                 | 1 / 175                                                                              |
| Newline chars                        | none (the `\n` seen in the CSV are *literal* backslash-n strings, not real newlines) |
| Rows containing `<...>` placeholders | 1,054 (e.g. `<name>`)                                                                |


**Impact on target — evidence:** length again flat across priorities; priority is driven by *content words* (incidents/outages/security language), not by how long the message is.


| priority | n      | mean chars | median chars |
| -------- | ------ | ---------- | ------------ |
| high     | 11,178 | 388.19     | 386.0        |
| medium   | 11,515 | 387.50     | 386.0        |
| low      | 5,894  | 385.06     | 384.0        |


**Correction strategies:** this is the main NLP input. Note the **escaped** `\n` and **placeholder tokens (**`<name>`**,** `<tel_num>`**)** → tokenizers will otherwise split them into junk tokens; normalising them (or treating placeholders as special tokens) is recommended at modelling time. Text is split roughly 57% English / 43% German.

### 3.3 `answer` — support-side reply (text)


| Measure                              | Value                                          |
| ------------------------------------ | ---------------------------------------------- |
| Non-null count                       | 28,580                                         |
| **Null count / %**                   | **7 / 0.02%**                                  |
| Unique                               | 28,580                                         |
| Char length mean / median            | 387.27 / 390.0                                 |
| Char length std / min / max          | 183.09 / 4 / 1,006                             |
| Word count mean / median             | 56.56 / 58.0                                   |
| Newline chars                        | none                                           |
| Rows containing `<...>` placeholders | 13,445 (≈47%, mostly email/phone placeholders) |


**Impact on target — evidence:** the `answer` is the *response*, which the model will not have at inference time for a fresh ticket. Length per priority is flat (mean ≈ 384–390). It has **no legitimate role as a feature** for priority prediction and is primarily useful as reference text or for a generation bench.

**Correction strategies:** exclude from priority features (leakage-ish / unavailable at prediction time). If kept for analysis, fill the 7 nulls from a template or drop them.

### 3.4 `type` — ticket category


| Type     | Count  | % of rows | High % within type | Low %  | Medium % |
| -------- | ------ | --------- | ------------------ | ------ | -------- |
| Incident | 11,466 | 40.11%    | 43.01%             | 18.86% | 38.14%   |
| Change   | 2,922  | 10.22%    | 40.01%             | 19.58% | 40.42%   |
| Problem  | 6,012  | 21.03%    | 39.11%             | 20.61% | 40.29%   |
| Request  | 8,187  | 28.64%    | 33.31%             | 23.46% | 43.23%   |


**Impact on target — evidence:**

- `Incident` skews **high**; `Request` skews **low**/medium — directionally intuitive.
- BUT the effect is modest: rn-high ranges only 33%→43%, and Cramér's V vs `priority` = **0.059** (weak). Alone it is a **weak classifier**.

**Correction strategies:** ordinal label encoding is a reasonable (not meaningful as numeric), better one-hot/embedding. Use as a *supporting* feature, not the main driver.

### 3.5 `queue` — routing queue (10 values)


| Queue                           | Count | % of rows | High %     | Low %  | Medium % |
| ------------------------------- | ----- | --------- | ---------- | ------ | -------- |
| Service Outages and Maintenance | 1,148 | 4.02%     | **70.99%** | 11.50% | 17.51%   |
| Technical Support               | 8,362 | 29.25%    | 58.60%     | 11.07% | 30.33%   |
| IT Support                      | 3,433 | 12.01%    | 48.85%     | 8.88%  | 42.27%   |
| Billing and Payments            | 2,788 | 9.75%     | 30.56%     | 21.59% | 47.85%   |
| Product Support                 | 5,252 | 18.37%    | 29.59%     | 20.14% | 50.27%   |
| Returns and Exchanges           | 1,437 | 5.03%     | 21.36%     | 36.95% | 41.68%   |
| Customer Service                | 4,268 | 14.93%    | 18.86%     | 35.07% | 46.06%   |
| Sales and Pre-Sales             | 918   | 3.21%     | 17.54%     | 36.17% | 46.30%   |
| General Inquiry                 | 405   | 1.42%     | 12.84%     | 55.56% | 31.60%   |
| Human Resources                 | 576   | 2.01%     | **9.55%**  | 49.65% | 40.80%   |


**Impact on target — evidence:**

- **Strongest single categorical driver**: Cramér's V = **0.286** (highest of all columns).
- Outage/Maintenance queues produce ~71% high-priority; HR/general-inquiry queues produce <13% high and ~50% low → a clear, almost monotonic gradient.

**Correction strategies:** near must-use feature. 10 levels, moderately imbalanced (29% in the largest) — encode as label/embedding; small categories (General Inquiry 1.4%) can be grouped or kept with care to avoid overfitting.

### 3.6 `language` — language of ticket


| Language | Count  | % of rows | High % | Low %  | Medium % |
| -------- | ------ | --------- | ------ | ------ | -------- |
| en       | 16,338 | 57.15%    | 38.84% | 20.65% | 40.51%   |
| de       | 12,249 | 42.85%    | 39.45% | 20.57% | 39.98%   |


**Impact on target — evidence:** distributions are **virtually identical** across en/de; Cramér's V = **0.006** → `language` has **no direct predictive value** for priority. It matters only for *tokenization* (German vs English preprocessing), not as a priority signal.

**Correction strategies:** do **not** impute missing values — there are none; use it only to select the tokenizer / pretrained model family (e.g. `multilingual` encoder).

### 3.7 `version` — version id (numeric-looking)


| Stats (numeric coercion) | Value                                                           |
| ------------------------ | --------------------------------------------------------------- |
| **Mean**                 | **278.38**                                                      |
| **Median**               | **400.0**                                                       |
| Std / min / max          | 165.96 / 51 / 400                                               |
| Q1 / Q3                  | 52 / 400                                                        |
| Raw values               | `400` → 18,599 (65.1%), `52` → 9,119 (31.9%), `51` → 869 (3.0%) |
| Nulls                    | 0                                                               |


**Priority impact (by value):**


| Version | High % | Medium % | Low %  |
| ------- | ------ | -------- | ------ |
| 400     | 39.30% | 40.08%   | 20.62% |
| 52      | 38.58% | 40.75%   | 20.67% |
| 51      | 40.28% | 39.70%   | 20.02% |



| priority | count  | mean   | median |
| -------- | ------ | ------ | ------ |
| high     | 11,178 | 279.55 | 400    |
| medium   | 11,515 | 277.24 | 400    |
| low      | 5,894  | 278.40 | 400    |


**Impact on target — evidence:** the priority composition is **identical across all three values** → Cramér's V = **0.006**. `version` is a near-constant, synthetic-looking attribute with **no predictive power**.

**Correction strategies:** safe to treat as categorical with the 3 raw levels, or **drop** it entirely — it cannot help discriminate priority. If retained, do not model 400/52/51 as numeric magnitudes (they are labels, and 400 is not "bigger" than 51 in a meaningful way).

### 3.8 Tags — `tag_1` … `tag_8` (multi-label, up to 8 per row)

**Structure:** each row carries an *ordered* list of topic tags (first tag most specific), topped out at 8. 136 unique tags overall; 134,165 total tag occurrences across 28,587 rows → **≈ 4.7 tags / row**; median row has 4 tags.


| # tags in row | Rows          |
| ------------- | ------------- |
| 1             | 13            |
| 2             | 123           |
| 3             | 2,922         |
| 4             | 10,984 (mode) |
| 5             | 8,671         |
| 6             | 3,834         |
| 7             | 1,475         |
| 8             | 565           |


**Per-slot missingness** (grows fast in the tail — the sparse columns carry little signal):


| Tag column | Nulls  | Null % | Unique |
| ---------- | ------ | ------ | ------ |
| `tag_1`    | 0      | 0.00%  | 116    |
| `tag_2`    | 13     | 0.05%  | 256    |
| `tag_3`    | 136    | 0.48%  | 392    |
| `tag_4`    | 3,058  | 10.70% | 554    |
| `tag_5`    | 14,042 | 49.12% | 602    |
| `tag_6`    | 22,713 | 79.45% | 575    |
| `tag_7`    | 26,547 | 92.86% | 427    |
| `tag_8`    | 28,022 | 98.02% | 224    |


**Top** `tag_1` **values (primary slot is always filled):** `Security` 5,880 · `Bug` 5,337 · `Feedback` 3,557 · `Feature` 3,081 · `Performance` 3,065 · `Billing` 1,382 · `Outage` 1,199 · `Network` 1,063 …

**Most-frequent tags overall (union of all slots):** `Tech Support` 16,505 · `IT` 16,325 · `Performance` 13,374 · `Feedback` 7,937 · `Bug` 7,492 · `Security` 7,046 · `Feature` 6,660 · `Documentation` 6,167 · `Disruption` 5,744 · `Outage` 4,364 · `Network` 3,693 · `Product` 2,503 · `Recovery` 2,494 · `Sales` 2,432 · `Hardware` 1,569 · `Billing` 1,534 · `Support` 1,401 · `Payment` 1,269 · `Marketing` 1,231 · `Maintenance` 1,162 · `Crash` 1,044 · `Software` 1,010 · `Virus` 826 · `Integration` 821 · `Account` 737.

**Impact on target — evidence (high-% share among tickets carrying that tag):**


| Tag           | Rows   | High %     | Low %  |
| ------------- | ------ | ---------- | ------ |
| Outage        | 4,364  | **54.03%** | 13.08% |
| Recovery      | 2,494  | 51.84%     | 14.03% |
| Maintenance   | 1,162  | 50.60%     | 17.38% |
| Disruption    | 5,744  | 48.83%     | 15.55% |
| Hardware      | 1,569  | 45.32%     | 17.08% |
| Network       | 3,693  | 43.92%     | 18.39% |
| Performance   | 13,374 | 43.10%     | 18.51% |
| Bug           | 7,492  | 41.78%     | 19.28% |
| IT            | 16,325 | 40.84%     | 20.02% |
| Tech Support  | 16,505 | 40.65%     | 19.82% |
| Security      | 7,046  | 39.60%     | 20.13% |
| Support       | 1,401  | 35.12%     | 20.99% |
| Feature       | 6,660  | 33.93%     | 22.30% |
| Documentation | 6,167  | 33.70%     | 23.29% |
| Feedback      | 7,937  | 33.39%     | 23.81% |
| Product       | 2,503  | 33.32%     | 21.85% |
| Sales         | 2,432  | 31.74%     | 25.21% |
| Marketing     | 1,231  | 31.19%     | 25.67% |
| Payment       | 1,269  | 30.94%     | 23.76% |
| Billing       | 1,534  | 29.34%     | 22.56% |


**Cramér's V of each tag slot vs** `priority`**:**


| Slot    | V vs priority |
| ------- | ------------- |
| `tag_1` | 0.155         |
| `tag_2` | 0.150         |
| `tag_3` | 0.168         |
| `tag_4` | **0.180**     |
| `tag_5` | 0.177         |
| `tag_6` | 0.161         |
| `tag_7` | 0.132         |
| `tag_8` | 0.091         |


**Takeaways:**

- Tags are **informative but secondary** to `queue`: outage/recovery/maintenance/disruption tags are strongly linked to **high** priority; request-ish tags (billing/sales/marketing/feature) sit **low**.
- Ordered first-slots (tag_1–tag_4) carry most of the signal; **tag_7/tag_8 are sparse (93–98% null) and only marginally informative**.

**Correction strategies:** treat as **multi-label** (binary presence vector or multi-hot over the 136-tag vocabulary), *not* as 8 independent ordinal slots. Imputing missing tags with `"unknown"` is fine; the sparse tail slots (`tag_6`–`tag_8`) may be dropped or folded into the multi-hot for efficiency.

---



## 4. Overall ranking of predictive value


| Rank | Column                        | Cramér's V vs `priority` | Verdict                                                                   |
| ---- | ----------------------------- | ------------------------ | ------------------------------------------------------------------------- |
| 1    | `queue`                       | **0.286**                | Strongest; use                                                            |
| 2    | `tag_4`                       | 0.180                    | Use (multi-label union better)                                            |
| 3    | `tag_5`                       | 0.177                    | Use                                                                       |
| 4    | `tag_3`                       | 0.168                    | Use                                                                       |
| 5    | `tag_6`                       | 0.161                    | Use (moderately sparse)                                                   |
| 6    | `tag_1`                       | 0.155                    | Use                                                                       |
| 7    | `tag_2`                       | 0.150                    | Use                                                                       |
| 8    | `tag_7`                       | 0.132                    | Optional (sparse)                                                         |
| 9    | `tag_8`                       | 0.091                    | Weak (98% null)                                                           |
| 10   | `type`                        | 0.059                    | Weak — supporting only                                                    |
| 11   | `language`                    | 0.006                    | No priority signal — preprocessing concern only                           |
| 12   | `version`                     | 0.006                    | No signal — near-constant, droppable                                      |
| –    | `subject` / `body` / `answer` | n/a (text)               | `subject`+`body` carry the content; `answer` is a response, not a feature |


**Bottom line:** priority is best explained by **routing queue + topic tags + the semantic content of subject/body**, with `answer`**,** `language`**, and** `version` **contributing nothing** as priority signals (answer being unavailable at inference time anyway). Missing data is concentrated in `subject` (13.4%) and the sparse tag tail — both cheap to handle via imputation from `body` (subject) and multi-hot + "unknown" (tags).