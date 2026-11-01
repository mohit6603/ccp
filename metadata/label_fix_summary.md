# Label fix summary

All 970 receipt labels (600 train, 370 test) were checked against their images and corrected where the JSON did not match what is printed.

- Original labels before this pass: the `Clean Dataset copy` folder on the original Mac, and this repository's history at commit `876dabf` (the last commit before the corrected labels were pushed).
- Every individual change, with old value, new value and reason: `label_fix_log.csv` (592 rows).
- All 970 files pass a schema check: same 8 keys in the same order, money values as plain strings, quantities as numbers.

## What changed

437 receipts were changed (279 train, 158 test). The other 533 matched their images and were left as they were.

| Field | Wrong value corrected | Was null, now filled |
|---|---|---|
| line_items | 169 | 168 |
| address | 84 | 0 |
| company | 19 | 0 |
| tax_amount | 2 | 20 |
| total | 10 | 0 |
| tax_number | 1 | 8 |
| date | 5 | 0 |
| service_charge | 0 | 4 |

Line items went from 2,226 to 2,781. Receipts with `line_items: null` dropped from 169 to 1.

## Labeling rules applied

- **Printed text wins.** Every value matches what is printed on the receipt. OCR misreads were fixed, for example O/0, I/1, S/5, B/8, "KP3" for "KPB" and "BELANGOR" for "SELANGOR". Fields not printed on the receipt stay null.
- **total** is the final amount payable after rounding, for example "TTL ATF RND", "Total Rounded", "Nett Total" or "Grand Total" after a coupon. When only TOTAL, rounding and CASH lines are printed, the printed TOTAL is kept.
- **tax_amount** is the GST amount as printed. A printed zero becomes "0.00", and negative GST on adjustment notes stays negative, for example "-0.27". It is null only when no tax amount is printed.
- **line_items** keep product rows in printed order.
  - Discount, promo, member-discount, deposit, rounding and payment rows are not items. Item amounts are the printed line amounts, before any separate discount rows.
  - Set or meal components and modifiers without their own printed amount are not separate items. Examples are fries and drinks in a meal, "Cold", "Less Ice" and "Medium Well".
  - Barcodes, stock codes, product codes and unit-of-measure codes (EA, PCS, @ SET, @ TIN, @ PKT) are not part of the description. Leading GST markers and trailing tax flags are also left out.
  - Names printed over several lines are joined with a space. Quantity and unit price are filled whenever they are printed.
  - A parking ticket gets one item, "PARKING FEE", with the fee as its amount.
- **address** includes every printed address line of the merchant header. For 99 Speed Mart, the branch line (for example "1237-TMN INDAH PERDANA") is part of the address on every receipt.
- **company**: the brand name versus legal entity choice was left as the original ground truth had it. Only misreads were fixed.
- **Partly hidden text** (punch holes, faded print, folded corners) was filled in only when the same merchant's other receipts make the value certain. Examples are a postcode, a GST number or a company name.

## Known limitations

- **receipt531**: the item names are blanked out on the image, so `line_items` stays null.
- **receipt948**: the Chinese item names are blanked out. Items are labeled from the visible English text. One row whose whole name is blanked was left out, so the item amounts add up to less than the total.
- **receipt663**: the day in the date is faded. It is labeled "28/04/2017", the most likely reading.
- **receipt580**: the GST number is kept as the merchant's real registration number, 000850247680, although one digit prints like a 6.
- 25 receipts have item amounts that add up to more than the total. All of them print a discount, promo, voucher or deposit row, which is expected under the rule above.
