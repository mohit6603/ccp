"""Compare model outputs with labels -> scores and error lists.

Usage (from the project folder):
    .venv/bin/python src/evaluate.py --split train --run train_baseline
    .venv/bin/python src/evaluate.py --split train --run trial_5 --partial   (only receipts the run attempted)

Reads   outputs/<run>/receiptN.json  and  <split>/receipt_labeled_text/receiptN.json
Writes  results/<run>/summary.csv      headline numbers
        results/<run>/per_field.csv    accuracy and null handling per field
        results/<run>/per_receipt.csv  every receipt x field: label, prediction, exact, fuzzy score
        results/<run>/errors.csv       only the mismatches
        results/<run>/line_items.csv   per receipt: label items, predicted items, correct items

Scoring rules (fixed before the first run, applied the same way to prediction and label):
- Text (company, address, tax_number, item description): trim, collapse whitespace, ignore case.
  Punctuation still counts. Fuzzy match = similarity >= 0.95 (rapidfuzz ratio).
- Overall accuracy (the headline agreed with the manager on 2026-10-31): the 7 scalar fields, where company,
  address and tax_number also count as correct when their letters and digits match after removing all
  spaces and punctuation ("27,JALAN" = "27, JALAN."). Every letter and digit must still be right.
  Date and amounts use the exact rule.
- Amounts (total, tax_amount, service_charge, unit_price, amount): compared as decimal numbers,
  so "9.000" equals "9.00"; thousands commas ignored; sign kept. Fuzzy equals exact.
- Date: parsed to a calendar date (day first, 2-digit years expanded, month names recognised);
  the raw-string match is reported too.
- Null only matches null. An empty string counts as a value.
- A receipt with no valid output file counts as wrong on every field.
- Line items: each label item is paired with at most one predicted item (best description
  similarity, amount agreement as tie-break). An item is correct when the description is a fuzzy
  match and quantity, unit_price and amount are all equal.
"""
import argparse
import csv
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rapidfuzz import fuzz

KEYS = ["company", "date", "address", "tax_number", "line_items", "total", "tax_amount", "service_charge"]
SCALAR = ["company", "date", "address", "tax_number", "total", "tax_amount", "service_charge"]
TEXT = {"company", "address", "tax_number"}
AMOUNT = {"total", "tax_amount", "service_charge"}
FUZZY_OK = 0.95
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def norm_text(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def to_decimal(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return Decimal(str(v).replace(",", "").strip())
    except InvalidOperation:
        return None


def parse_date(v):
    """Day-first calendar date from the formats seen in the labels; None if not parseable."""
    s = str(v).upper().strip()
    if re.fullmatch(r"\d{8}", s):  # no separators: 20180428 (year first) or 25032018 (day first)
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}" if s[:2] in ("19", "20") and 1 <= int(s[4:6]) <= 12 else f"{s[:2]}/{s[2:4]}/{s[4:]}"
    elif re.fullmatch(r"\d{6}", s):  # 250318
        s = f"{s[:2]}/{s[2:4]}/{s[4:]}"
    parts = re.findall(r"\d+|[A-Z]+", s)
    nums = [p for p in parts if p.isdigit()]
    mon = next((MONTHS[p[:3]] for p in parts if p[:3] in MONTHS), None)
    try:
        if mon is not None and len(nums) >= 2:  # 11 MAY 2018, 27/MAR/2018, 28-FEB-2018
            d, y = int(nums[0]), int(nums[1])
        elif len(nums) >= 3 and len(nums[0]) == 4:  # 2018-05-01, 2018/03/27
            y, mon, d = int(nums[0]), int(nums[1]), int(nums[2])
        elif len(nums) >= 3:  # 24/05/2017, 25-04-18; month-first only when day-first is impossible (4/22/2018)
            d, mon, y = int(nums[0]), int(nums[1]), int(nums[2])
            if mon > 12 >= d:
                d, mon = mon, d
        else:
            return None
        if y < 100:
            y += 2000
        return date(y, mon, d)
    except ValueError:
        return None


def norm_alnum(v):
    """Letters and digits only (any script), lower case: '27,JALAN' and '27, JALAN.' both give '27jalan'."""
    return re.sub(r"[\W_]+", "", str(v)).lower()


def similarity(a, b):
    return fuzz.ratio(norm_text(a), norm_text(b)) / 100


def compare_scalar(field, lab, pred):
    """Return (exact, fuzzy_score) for two non-null values."""
    if field in AMOUNT:
        ok = to_decimal(lab) is not None and to_decimal(lab) == to_decimal(pred)
        return ok, float(ok)
    if field == "date":
        dl, dp = parse_date(lab), parse_date(pred)
        ok = (dl == dp) if (dl and dp) else norm_text(lab) == norm_text(pred)
        return ok, 1.0 if ok else similarity(lab, pred)
    sim = similarity(lab, pred)
    return norm_text(lab) == norm_text(pred), sim


def num_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return to_decimal(a) is not None and to_decimal(a) == to_decimal(b)


def match_items(lab_items, pred_items):
    """Greedy one-to-one pairing; returns list of (label_item, pred_item_or_None, correct)."""
    cands = []
    for i, li in enumerate(lab_items):
        for j, pi in enumerate(pred_items):
            sim = similarity(li.get("description", ""), pi.get("description", ""))
            amt = num_equal(li.get("amount"), pi.get("amount"))
            if sim >= 0.5 or amt:
                cands.append((sim + (0.3 if amt else 0), i, j))
    used_l, used_p, pairs = set(), set(), {}
    for _, i, j in sorted(cands, reverse=True):
        if i not in used_l and j not in used_p:
            used_l.add(i), used_p.add(j)
            pairs[i] = j
    out = []
    for i, li in enumerate(lab_items):
        pi = pred_items[pairs[i]] if i in pairs else None
        ok = pi is not None and similarity(li.get("description", ""), pi.get("description", "")) >= FUZZY_OK \
            and all(num_equal(li.get(k), pi.get(k)) for k in ("quantity", "unit_price", "amount"))
        out.append((li, pi, ok))
    return out


def schema_ok(pred):
    """Same value types as the labels: strings or null for scalars, list or null for items."""
    for k in SCALAR:
        if pred[k] is not None and not isinstance(pred[k], str):
            return False
    items = pred["line_items"]
    if items is None:
        return True
    for it in items:
        q = it["quantity"]
        if not isinstance(it["description"], str) or (q is not None and (isinstance(q, bool) or not isinstance(q, (int, float)))):
            return False
        if any(it[k] is not None and not isinstance(it[k], str) for k in ("unit_price", "amount")):
            return False
    return True


def show(v):
    return "null" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred-dir", help="override the prediction folder (default outputs/<run>)")
    ap.add_argument("--partial", action="store_true",
                    help="score only receipts the run attempted (raw/receiptN.txt exists); for trial runs")
    ap.add_argument("--names-file", help="score only the receipts listed in this file, e.g. src/dev100.txt")
    ap.add_argument("--res-dir", help="override the results folder (default results/<run>)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_dir = Path(args.pred_dir) if args.pred_dir else root / "outputs" / args.run
    res_dir = Path(args.res_dir) if args.res_dir else root / "results" / args.run
    res_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted((root / args.split / "receipt_labeled_text").glob("*.json"), key=lambda p: int(p.stem[7:]))
    if args.partial:
        labels = [p for p in labels if (pred_dir / "raw" / f"{p.stem}.txt").exists()]
    if args.names_file:
        keep = set((root / args.names_file).read_text().split())
        labels = [p for p in labels if p.stem in keep]

    per_field = {f: dict(n=0, exact=0, overall=0, fuzzy=0, sim_sum=0.0, sim_n=0, both_null=0, hallucinated=0, missed=0,
                         label_null=0, label_value=0)
                 for f in SCALAR + ["line_items"]}
    rows, item_rows = [], []
    n_valid = n_schema = all_correct = wrong_fields_total = 0
    tp = n_lab_items = n_pred_items = 0
    attr = {k: [0, 0] for k in ("description", "quantity", "unit_price", "amount")}

    for lp in labels:
        name = lp.stem
        lab = json.loads(lp.read_text())
        pp = pred_dir / f"{name}.json"
        pred = json.loads(pp.read_text()) if pp.exists() else None
        valid = pred is not None
        n_valid += valid
        if valid and schema_ok(pred):
            n_schema += 1
        wrong = 0
        for f in SCALAR:
            st = per_field[f]
            st["n"] += 1
            lv = lab[f]
            pv = pred[f] if valid else "<no valid output>"
            st["label_null" if lv is None else "label_value"] += 1
            if not valid:
                exact, score = False, 0.0
            elif lv is None and pv is None:
                exact, score = True, 1.0
                st["both_null"] += 1
            elif lv is None:
                exact, score = False, 0.0
                st["hallucinated"] += 1
            elif pv is None:
                exact, score = False, 0.0
                st["missed"] += 1
            else:
                exact, score = compare_scalar(f, lv, pv)
                st["sim_sum"] += score
                st["sim_n"] += 1
            fuzzy_ok = exact or (f in TEXT and score >= FUZZY_OK) or (f == "date" and exact)
            overall_ok = exact or (valid and f in TEXT and lv is not None and pv is not None
                                   and norm_alnum(lv) != "" and norm_alnum(lv) == norm_alnum(pv))
            st["exact"] += exact
            st["overall"] += overall_ok
            st["fuzzy"] += fuzzy_ok
            wrong += not exact
            raw_equal = valid and lv is not None and isinstance(pv, str) and norm_text(lv) == norm_text(pv)
            rows.append([name, f, show(lv), show(pv), int(exact), f"{score:.3f}", int(raw_equal) if f == "date" else "",
                         int(overall_ok)])

        # line items
        st = per_field["line_items"]
        st["n"] += 1
        li = lab["line_items"]
        pi = pred["line_items"] if valid else None
        st["label_null" if li is None else "label_value"] += 1
        lab_items, pred_items = li or [], (pi if isinstance(pi, list) else []) if valid else []
        pairs = match_items(lab_items, pred_items)
        correct = sum(ok for _, _, ok in pairs)
        for l_it, p_it, _ in pairs:
            if p_it is not None:
                attr["description"][0] += similarity(l_it["description"], p_it.get("description", "")) >= FUZZY_OK
                for k in ("quantity", "unit_price", "amount"):
                    attr[k][0] += num_equal(l_it.get(k), p_it.get(k))
                for k in attr:
                    attr[k][1] += 1
        tp += correct
        n_lab_items += len(lab_items)
        n_pred_items += len(pred_items)
        if not valid:
            items_exact = False
        elif li is None and pi is None:
            items_exact = True
            st["both_null"] += 1
        elif li is None:
            items_exact = False
            st["hallucinated"] += 1
        elif pi is None:
            items_exact = False
            st["missed"] += 1
        else:
            items_exact = correct == len(lab_items) == len(pred_items)
        st["exact"] += items_exact
        st["overall"] += items_exact
        st["fuzzy"] += items_exact
        wrong += not items_exact
        item_rows.append([name, len(lab_items), len(pred_items) if valid else "", correct, int(items_exact)])
        if not items_exact:
            rows.append([name, "line_items", show(li), show(pi) if valid else "<no valid output>", 0, "", "", 0])
        else:
            rows.append([name, "line_items", show(li), show(pi), 1, "", "", 1])

        wrong_fields_total += wrong
        all_correct += wrong == 0

    n = len(labels)
    prec = tp / n_pred_items if n_pred_items else 0.0
    rec = tp / n_lab_items if n_lab_items else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    macro = sum(per_field[f]["exact"] / n for f in SCALAR) / len(SCALAR)
    macro_overall = sum(per_field[f]["overall"] / n for f in SCALAR) / len(SCALAR)
    macro_fuzzy = sum(per_field[f]["fuzzy"] / n for f in SCALAR) / len(SCALAR)

    summary = [
        ("receipts", n),
        ("valid_json_pct", 100 * n_valid / n),
        ("schema_ok_pct", 100 * n_schema / n),
        ("overall_accuracy_pct (7 scalar fields, spaces and punctuation ignored)", 100 * macro_overall),
        ("macro_exact_accuracy_pct (7 scalar fields)", 100 * macro),
        ("macro_fuzzy_accuracy_pct (7 scalar fields, text >= 95% similar)", 100 * macro_fuzzy),
        ("receipts_all_fields_correct_pct", 100 * all_correct / n),
        ("avg_wrong_fields_per_receipt", wrong_fields_total / n),
        ("line_items_precision_pct", 100 * prec),
        ("line_items_recall_pct", 100 * rec),
        ("line_items_f1_pct", 100 * f1),
        ("line_items_receipt_exact_pct", 100 * per_field["line_items"]["exact"] / n),
    ] + [(f"matched_item_{k}_accuracy_pct", 100 * c / t if t else 0.0) for k, (c, t) in attr.items()]

    with open(res_dir / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in summary:
            w.writerow([k, f"{v:.2f}" if isinstance(v, float) else v])
    with open(res_dir / "per_field.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "exact_pct", "overall_pct (spaces and punctuation ignored)", "fuzzy_pct",
                    "mean_similarity_when_both_present", "label_null", "label_value", "correct_null",
                    "hallucinated (label null, model gave value)", "missed (label has value, model gave null)"])
        for f in SCALAR + ["line_items"]:
            st = per_field[f]
            w.writerow([f, f"{100 * st['exact'] / n:.2f}", f"{100 * st['overall'] / n:.2f}", f"{100 * st['fuzzy'] / n:.2f}",
                        f"{st['sim_sum'] / st['sim_n']:.3f}" if st["sim_n"] else "",
                        st["label_null"], st["label_value"], st["both_null"], st["hallucinated"], st["missed"]])
    with open(res_dir / "per_receipt.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["receipt", "field", "label", "prediction", "exact", "fuzzy_score", "date_raw_string_match", "overall_match"])
        w.writerows(rows)
    with open(res_dir / "errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["receipt", "field", "label", "prediction", "fuzzy_score", "overall_match (spaces and punctuation ignored)"])
        w.writerows([r[0], r[1], r[2], r[3], r[5], r[7]] for r in rows if r[4] == 0)
    with open(res_dir / "line_items.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["receipt", "label_items", "predicted_items", "correct_items", "list_exact"])
        w.writerows(item_rows)

    print(f"run={args.run} split={args.split} receipts={n}")
    for k, v in summary:
        print(f"  {k:45s} {v:.2f}" if isinstance(v, float) else f"  {k:45s} {v}")
    print("  per field exact%: " + ", ".join(f"{f}={100 * per_field[f]['exact'] / n:.1f}" for f in SCALAR + ["line_items"]))
    print("  per field overall%: " + ", ".join(f"{f}={100 * per_field[f]['overall'] / n:.1f}" for f in SCALAR))
    print(f"results written to {res_dir}")


if __name__ == "__main__":
    main()
