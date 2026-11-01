# 10. Questions your manager may ask

Short, honest answers. The details are in the chapters mentioned.

**What does 96.22% mean?**
On 370 receipts the model had never seen, 96.22% of the single-value fields are correct: company, date, address, tax number, total, tax amount and service charge. For text fields, spaces and punctuation are ignored, but every letter and digit must be right. Under strict matching, where punctuation counts, the number is 92.47%. Line items score 83.42% F1. See chapter 7.

**Was the test fair?**
Yes:
- The model was never taught with any test receipt.
- Near-duplicates of test receipts were removed from the teaching data.
- All choices were made on train receipts.
- The test was run once with the setup frozen, and nothing was changed afterwards, not even the one broken answer.

See chapters 1.4 and 7.

**How sure are we about the number?**
Redrawing the test receipts 20,000 times gives a 95% range of 95.3% to 97.1% for overall accuracy, and 91.4% to 93.5% for strict accuracy. Both ranges are above 90%. See chapter 7.4.

**Does every receipt come out fully correct?**
No. 78% of test receipts have all 7 single-value fields right. 62% also have every line item right. One receipt in 370 gave broken output. For fully automatic processing, a human check of uncertain receipts would still be needed. See chapter 7.5.

**Will it work on our own receipts?**
That is not known yet. The model was tested on Malaysian shop receipts, mostly scanned. Other countries' formats (for example Indian GST bills with 15-character GSTINs), phone photos, handwriting or other languages were not measured, and accuracy will probably be lower, especially for tax numbers and line items. The way to find out: label 30 to 50 of our receipts and score them with the same scripts. If needed, the model can then be fine-tuned on our receipts in the same way. See chapter 7.9.

**Does any data leave the computer?**
No. The model runs entirely on the laptop, and no receipt is sent to an online AI service or API. The only download was the model itself, from the official Qwen page. The project's code and the receipts (a public research dataset) are in a GitHub repository. It is currently public, which was already the case before this work. See the README's Data handling section.

**What does it cost?**
Nothing beyond the hardware. The model is free (Apache 2.0 licence), and all the software is free. It runs on a MacBook Air.

**How fast is it?**
About 21 seconds per receipt on the MacBook Air M5. The 370 test receipts took 2.2 hours. Fine-tuning took about 7 hours, once.

**Can it run on our office laptops?**
It depends on the laptop:
- **An Apple Silicon Mac (M1 or newer):** everything runs, at about the same speed.
- **An Ubuntu laptop without a graphics card:** the scoring and all the checks run, and the model itself runs too, but very slowly, because it uses only the processor. Chapter 8 has the measured speed and the options.
- **Fine-tuning** needs an Apple Silicon Mac with 16 GB of memory, or a GPU machine.

**What are the weak points?**
- **Address** is the weakest field: 87.8% overall and 64.3% exact. It is mostly spacing and punctuation, partly because the labels themselves are inconsistent.
- **Line items** are hard: 83% F1.
- **Broken output:** one answer in 370 was not valid JSON.
- **Other receipt types:** not measured.

**What would improve it further?**
- **Our own receipts:** label some of our own receipts and fine-tune on them; this matters most if our receipts look different.
- **More examples:** add more teaching examples of the weak cases, such as long item lists and unusual layouts.
- **A retry for broken output:** an automatic check that asks again, or flags the receipt for a person, when the output is not valid JSON.

**Can we trust the labels it was measured against?**
All 970 labels were checked against their images before any model was run, and 437 receipts were corrected. Every change is recorded in `metadata/label_fix_log.csv`, and the original labels are kept in the git history. See chapter 1.5.

**Can someone else reproduce this?**
Yes. Everything is described in chapter 8 and the README:
- **The exact environment:** `requirements.txt` was tested by building a fresh environment from it, and it gives identical results.
- **The model files:** checksums prove they are the same.
- **The scores:** they can be recomputed on any computer.
