# 5. Step 3: the first results and better instructions

## 5.1 The baseline: the model as it is

The first run used the model exactly as downloaded, with the simple instruction in `src/prompt_v1.txt` and no cleanup. A first run like this is called the **baseline**: every improvement is measured against it.

```bash
.venv/bin/python src/run_model.py --split train --run train_baseline --no-clean
.venv/bin/python src/evaluate.py --split train --run train_baseline
```

It ran on all 600 train receipts in 3.7 hours, about 22 seconds per receipt.

| Measure | Result |
|---|---|
| Valid JSON | 99.83% (599 of 600) |
| **Overall accuracy** | **90.02%** |
| Exact accuracy | 86.64% |
| Line-item F1 | 36.11% |

Exact accuracy per field:

| company | date | address | tax_number | total | tax_amount | service_charge |
|---|---|---|---|---|---|---|
| 84.8% | 97.3% | 60.2% | 81.8% | 92.7% | 93.0% | 96.7% |

## 5.2 What went wrong

We read `results/train_baseline/errors.csv` and looked at the images of wrong receipts. The same kinds of mistakes kept coming back:

| Mistake | Example | How often |
|---|---|---|
| **Another number taken as the tax number**: a company registration number, a phone number, or a reference number | receipt2: the number on the "Room No" line | 16 hallucinated tax numbers |
| **Inventing a tax amount**, often by taking the rounding adjustment as tax | receipt2: rounding -0.01 given as tax_amount | 11 hallucinated tax amounts |
| **Inventing a service charge** | | 15 hallucinated service charges |
| **The total before rounding** instead of the final amount paid | receipt2: 60.31 instead of 60.30 | |
| **Product codes and quantities left in item names** | receipt2: "... \<i> 62483" | many |
| **Calculating a unit price** that is not printed | | many |
| **Text format differences**: spaces, a "GST REG #" prefix before the number, "RM" in front of amounts | | many |

Receipt2's baseline answer, compared with its label:

| Field | Label | Baseline answer | Result |
|---|---|---|---|
| company | INDAH GIFT & HOME DECO | same | ✓ |
| date | 19/10/2018 | same | ✓ |
| address | 27, JALAN DEDAP 13, ... | 27,JALAN DEDAP 13, ... | ✓ overall rule, ✗ exact rule |
| tax_number | null | 050100035279 | ✗ hallucination |
| line_items | 2 items | 2 items, 1 correct | ✗ |
| total | 60.30 | 60.31 | ✗ before rounding |
| tax_amount | null | -0.01 | ✗ rounding taken as tax |
| service_charge | null | null | ✓ |

## 5.3 Prompt v2: clearer rules

Based on these mistakes, a new instruction was written: `src/prompt_v2.txt`. It is 621 words; v1 was 245. It tells the model exactly what each field is, and what it is not:

- **company:** prefer the registered name (ending in SDN BHD, BHD, S/B, ENTERPRISE or TRADING) and leave out the registration number.
- **date:** as printed, without the time.
- **address:** only the header address lines, without phone, fax, email, website or registration numbers.
- **tax_number:** only the GST number, usually 12 digits, without its label. It says clearly that a registration number, a phone or fax number, or an invoice, table, room or member number is **not** a tax number, and that the answer is null if none is printed.
- **line_items:**
  - the product name only, without codes, unit codes, tax flags, a leading quantity or "@ price";
  - the unit price only if it is printed, and never calculated;
  - discount, rounding, subtotal, payment and set-component rows are not items.
- **total:** the final amount after rounding, with an example (7.72 with rounding -0.02 gives 7.70); never the subtotal, the cash or the change.
- **tax_amount:** "0.00" if printed as zero; rounding is not tax; null if not printed.
- **service_charge:** null if not printed.
- **General rules:** read easily confused characters carefully (0/O, 1/I, 5/S, 8/B), and write amounts without currency signs.

At the same time, the **cleanup step** was added (chapter 3.4). It fixes format differences such as "RM" before amounts or "GST REG #" before the tax number.

## 5.4 The result on dev100

To compare fairly, both versions are scored on the same 100 dev receipts:

| Version | Overall | Exact | 95% similar | Line-item F1 |
|---|---|---|---|---|
| Baseline (prompt v1, no cleanup) | 91.00% | 87.57% | 93.43% | 31.00% |
| Prompt v2 + cleanup | 91.86% | 87.86% | 94.43% | **47.99%** |

- **Line items improved a lot,** from 31% to 48% F1.
- **The single-value fields barely moved,** from 91.00% to 91.86% overall.

Receipt2 with prompt v2:
- **Fixed:** the total is right (60.30), tax_amount is null, and the product code is gone, so both items are now correct.
- **Still wrong:** the model again took the "Room No" number as the tax number, even though the new instruction says room numbers are not tax numbers.

## 5.5 Why we moved to fine-tuning

The model kept some habits no matter how clearly the instruction was written. It liked to fill in a tax number whenever it saw a 12-digit number, for example. Writing ever-longer rules gave smaller and smaller gains.

**Fine-tuning** works differently. Instead of describing the rules in words, we **show** the model hundreds of receipts together with their correct answers. Then it learns the labelling conventions directly from the examples. That is the next chapter.
