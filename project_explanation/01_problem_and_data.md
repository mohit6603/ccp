# 1. The problem and the data

## 1.1 What we want the model to do

We give the model a picture of a receipt. It must give back a JSON object with 8 fields.

**JSON** is a simple text format made of `"name": value` pairs. Computers read it easily, and people can read it too. In JSON, `null` means "empty", which here means "this information is not on the receipt".

Here is the label (the correct answer) for receipt2:

```json
{
    "company": "INDAH GIFT & HOME DECO",
    "date": "19/10/2018",
    "address": "27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU, JOHOR.",
    "tax_number": null,
    "line_items": [
        {"description": "ST-PRIVILEGE CARD/GD INDAH", "quantity": 1, "unit_price": "10.00", "amount": "10.00"},
        {"description": "GF-TABLE LAMP/STITCH <I>",   "quantity": 1, "unit_price": "55.90", "amount": "55.90"}
    ],
    "total": "60.30",
    "tax_amount": null,
    "service_charge": null
}
```

## 1.2 The 8 fields

| Field | Meaning | receipt2 |
|---|---|---|
| company | the shop's name at the top of the receipt | `INDAH GIFT & HOME DECO` |
| date | the date of the purchase, written exactly as printed | `19/10/2018` |
| address | the shop's address lines from the header, joined into one line | `27, JALAN DEDAP 13, ...` |
| tax_number | the shop's GST registration number, which in Malaysia is usually 12 digits | `null`, because no GST number is printed |
| line_items | the list of products bought | 2 items |
| total | the final amount paid, after rounding | `60.30` |
| tax_amount | the GST tax amount, if printed | `null` |
| service_charge | the service charge, if printed (mostly at restaurants) | `null` |

Each line item has 4 parts:

| Part | Meaning |
|---|---|
| description | the product name, as printed |
| quantity | how many were bought (a number) |
| unit_price | the price of one unit, but only if it is printed separately |
| amount | the price on that line |

Amounts are written as text in quotes, like `"60.30"`, so that the exact digits are kept.

**Why receipt2's total is 60.30 and not 65.90.** The two items add up to 65.90. But the receipt also prints a 10% discount of -5.59, which gives 60.31, and then a rounding adjustment of -0.01. So the customer paid 60.30. The discount line is not an item, and the rounding is not tax.

## 1.3 Where the receipts come from

- **Source:** the receipts come from **SROIE**, a public collection of scanned Malaysian shop receipts made for a research competition (ICDAR 2019).
- **This dataset:** the project owner picked 970 of them, split them into train and test, and added four fields that SROIE does not have: tax_number, line_items, tax_amount and service_charge.
- **File names:** each image and its label share the same name. `receipt2.jpg` belongs with `receipt2.json`.

```
train/receipt_images/receipt2.jpg          the picture
train/receipt_labeled_text/receipt2.json   the correct answer (label)
```

`metadata/image_mapping.csv` links each receipt to its original SROIE file name.

## 1.4 Three groups of receipts, and why

Think of preparing for an exam:

| Group | Exam comparison | Size | How it is used |
|---|---|---|---|
| **train** | practice questions with answers | 600 | to find mistakes, improve the instructions, and teach the model |
| **dev100** | a mock exam made from practice questions, but never used for studying | 100 of the 600 train receipts (every 6th one) | to compare versions of the model and pick the best one |
| **test** | the final exam | 370 | used **once**, at the very end |

**Why the test is used only once.** Suppose we checked the test score after every change and kept the changes that raised it. Then we would slowly tune the model to those 370 receipts, and the score would look better than the model really is on new receipts. So all choices were made on train and dev100. The test was run once, and nothing was changed after that.

**Why dev100 is needed.** We still need a fair way to compare versions, for example "is the model better after 1 or 2 rounds of teaching?". dev100 is a fair judge, because the model is never taught with those 100 receipts. The list of names is in `src/dev100.txt`.

## 1.5 Fixing the labels first

If a label is wrong, a correct answer from the model gets marked as a mistake. So before any model was run, on 2026-10-30, all 970 labels were checked against their images.

| | |
|---|---|
| Receipts corrected | 437 of 970 (279 train, 158 test) |
| Individual changes | 592 |
| Line items | 2,226 before, 2,781 after (many items were missing) |
| Receipts with no item list | 169 before, 1 after |

The main rules used:
- **Printed text wins.** Misread characters were fixed, for example O and 0, I and 1, S and 5, B and 8.
- **total** is the final amount after rounding.
- **Not items:** discount, rounding, deposit and payment rows are not items, and neither are parts of a set meal.
- **Item names:** product codes, barcodes and unit codes (EA, PCS) are not part of the name.
- **Empty fields:** a field that is not printed stays null.

Where to find the record:
- **Every change, with the old value, the new value and the reason:** `metadata/label_fix_log.csv`.
- **The full list of rules:** `metadata/label_fix_summary.md`.
- **The original labels:** still in the GitHub history, at commit `876dabf`.

The labels were fixed **before** the model was ever run, so the model's answers could not influence them.

## 1.6 Near-duplicate receipts

Three train receipts are almost the same as three test receipts:
- receipt443 is the same receipt as test receipt442.
- receipt796 is the same receipt as test receipt249.
- receipt729 and test receipt746 are consecutive receipts with identical content.

They stay in the dataset. But the three train receipts were **left out of the teaching data**. Otherwise the model could memorise a receipt and then "recognise" it in the test, which would make the test score look better than it should.

## 1.7 What the data looks like in numbers

How often each field is empty (null) in the 600 corrected train labels:

| Field | Empty in |
|---|---|
| company, date, total | 0% |
| address, line_items | 0.2% (1 receipt) |
| tax_amount | 3.7% |
| tax_number | 4.7% |
| service_charge | 93.8% |

- **Items:** there are 1,697 line items in train, 2.8 per receipt on average. 380 of them have no separate unit price.
- **Image sizes:** from 463 × 605 pixels (receipt6) up to 4961 × 7016 pixels (receipt956).

**Why this matters.** service_charge is empty on almost every receipt. A model that invents a service charge would be wrong very often. So handling empty fields correctly is just as important as reading text correctly.

## 1.8 Known problems in a few receipts

- receipt531 and receipt948: the item names are blanked out in the image.
- receipt663: the day in the date is faded.
- receipt580: one digit of the GST number is unclear.
