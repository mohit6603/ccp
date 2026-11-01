# 4. Step 2: scoring (`src/evaluate.py`)

## 4.1 What it does

The script compares each answer in `outputs/<run>/` with its label, field by field. It writes the scores to `results/<run>/`.

```bash
.venv/bin/python src/evaluate.py --split test --run test_final
.venv/bin/python src/evaluate.py --split train --run dev100_lora_v1_ep2 --names-file src/dev100.txt
```

- **`--names-file src/dev100.txt`** scores only the dev100 receipts.
- **`--res-dir FOLDER`** writes the scores to another folder, so an existing `results/<run>/` is not overwritten.
- **`--partial`** scores only the receipts a run has finished so far. It is useful during a run.

The rules below were fixed **before** the first run. They are applied the same way to every run, and the same way to the answer and the label. That way nobody can "move the goalposts" to make a result look better.

One rule was added later: the **overall** rule, which became the headline.
- **When:** it was agreed with the manager on 2026-10-31.
- **What it saw:** it was chosen after the first results without fine-tuning were seen, but before any result of the fine-tuned model existed.
- **Where it applies:** to every run, including the very first one.

## 4.2 Three ways to judge a text field

The text fields are company, address and tax_number. Let's take receipt2's address. The printed receipt says `27,JALAN DEDAP 13,` with no space after the first comma, but the label puts spaces after the commas:

```
label:  27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU, JOHOR.
model:  27,JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU,JOHOR.
```

| Rule | How it compares | receipt2's address |
|---|---|---|
| **Exact** | Ignore capital letters and extra spaces, but every comma, full stop and needed space counts | ✗ wrong: "27, J" vs "27,J" |
| **Overall** (the headline) | Remove **all** spaces and punctuation, then compare: `27jalandedap13tamanjohorjaya81100johorbahrujohor`. Every letter and digit must still be right. | ✓ correct |
| **95% similar** | Correct if the two texts are at least 95% similar (a standard similarity score, from a package called rapidfuzz) | ✓ correct (98.4% similar) |

**Why the overall rule is the headline.** The labels are not consistent about spaces after commas. Some write "27,JALAN" and some "27, JALAN". A spacing difference is not a reading mistake, so the overall rule ignores spaces and punctuation, but it still demands every letter and digit.

The 95%-similar rule is more forgiving. It also lets one or two wrong letters pass in a long text, so it is reported only as a secondary number.

## 4.3 Amounts and dates

**Amounts** (total, tax_amount, service_charge, and each item's unit_price and amount) are compared as numbers:
- `"9.000"` equals `"9.00"`, and `"1,234.50"` equals `"1234.50"`;
- `"60.31"` does not equal `"60.30"`. A wrong digit is always wrong; there is no "similar" for amounts.

**Dates** are turned into calendar dates first, reading the **day first** (the Malaysian style). So all of these are the same date:

```
19/10/2018   19-10-18   19 OCT 2018   2018-10-19   19102018
```

- **Two-digit years:** "18" becomes 2018.
- **Month-first:** a date like 4/22/2018 is read month-first only because day-first is impossible (there is no month 22).

## 4.4 Empty fields (null)

| Label | Answer | Result | Name |
|---|---|---|---|
| null | null | correct | |
| null | a value | wrong | **hallucination**: the model invented something |
| a value | null | wrong | **miss**: the model left out something printed |
| a value | a value | compare with the rules above | |

**Example:** the model as it is gave receipt2 the tax_number `050100035279`. On the receipt, that number is printed on the "Room No" line; it is not a GST number. The label is null, so this counts as a hallucination.

## 4.5 Line items

Line items are the hardest part, because the answer can have a different number of items than the label, in any order.

**Step 1: pair the items.**
- Each label item is paired with at most one answer item: the one with the most similar description.
- A pair is only considered if the descriptions are at least 50% similar, or the amounts are equal. Equal amounts also give the pair a bonus.
- Pairs are chosen greedily: the best-matching pairs are taken first.

**Step 2: judge each pair.** An item is **correct** when all of these hold:
- its description is at least **95% similar**;
- its quantity, unit price and amount are **all equal**, with null only matching null.

**Step 3: count.**

| Measure | Formula | Meaning |
|---|---|---|
| **Precision** | correct items ÷ items the model wrote | how much of what the model wrote is right |
| **Recall** | correct items ÷ items in the labels | how much of what is printed the model got |
| **F1** | 2 × precision × recall ÷ (precision + recall) | one number that balances both |

A small example: the label has 4 items. The model writes 5 items, and 3 of them are correct.
- precision = 3/5 = 60%
- recall = 3/4 = 75%
- F1 = 2 × 0.60 × 0.75 ÷ 1.35 = **66.7%**

**Receipt2 with the model as it is:**

| Label item | Model item | Similarity | Result |
|---|---|---|---|
| ST-PRIVILEGE CARD/GD INDAH | ST-PRIVILEGE CARD/**B**D INDAH | 96.2% | correct (≥ 95%, all amounts equal) |
| GF-TABLE LAMP/STITCH \<I> | GF-TABLE LAMP/STITCH \<i> **62483** | 88.9% | wrong (below 95%: the product code was left in) |

So 1 of 2 items is correct, and the receipt's item list is not "fully right".

**"Every line item right"** for a receipt means every label item is correct **and** the model wrote no extra items.

## 4.6 Numbers for whole receipts

| Measure | Meaning |
|---|---|
| Valid JSON | the share of receipts whose answer could be read and had the right keys |
| Overall, exact and 95%-similar accuracy | the average over the 7 single-value fields (everything except line_items) and over all receipts |
| All 8 fields right | the share of receipts where all 7 single-value fields are exactly right **and** every line item is right. This is the strictest measure. |
| Average wrong fields per receipt | how many of the 8 fields are wrong, on average |

**An invalid answer** (broken JSON) counts as wrong on **every** field. The scorer never repairs a broken answer.

## 4.7 The score files

Every run gets five files in `results/<run>/`. They are CSV files, which open in Excel or Google Sheets.

| File | What is in it | Use it to |
|---|---|---|
| `summary.csv` | the headline numbers | report results |
| `per_field.csv` | for each field: exact, overall and 95%-similar accuracy, hallucinations and misses | find the weakest fields |
| `per_receipt.csv` | every receipt × every field: label, answer, and whether it was right under each rule | look at one receipt |
| `errors.csv` | only the wrong ones | see the mistakes quickly |
| `line_items.csv` | per receipt: items in the label, items in the answer, correct items | check line items |

**Receipt2 in `results/dev100_lora_v1_ep2/per_receipt.csv`** (the fine-tuned model):

| Field | Exact | Overall |
|---|---|---|
| company | 1 | 1 |
| date | 1 | 1 |
| address | 0 | 1 |
| tax_number | 1 | 1 |
| total | 1 | 1 |
| tax_amount | 1 | 1 |
| service_charge | 1 | 1 |
| line_items | 1 | 1 |

(1 means right, 0 means wrong.) Only the address differs, and only in the spaces after commas.

## 4.8 A self-test of the scorer

To check that the scorer itself has no bugs, the labels were scored against themselves. Every measure came out at exactly 100%, as it should.
