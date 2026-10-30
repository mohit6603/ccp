"""Select receipts for the Phase 0 label audit (fixed seed) -> label_audit/audit_sample.csv

Sample: all needs_review=yes, 40 generated_from_ocr_text and 20 ground_truth (half from each split).
"""
import csv
import random

random.seed(42)
split = {r["name"]: r["split"] for r in csv.DictReader(open("split_mapping.csv"))}
rows = list(csv.DictReader(open("labeled_text_sources.csv")))
for r in rows:
    r["name"] = r["json_file"][:-5]

review = [r for r in rows if r["needs_review"] == "yes"]
rest = [r for r in rows if r["needs_review"] != "yes"]


def stratified(source, per_split):
    """per_split random receipts from each of train and test, so both splits are audited."""
    pool = [r for r in rest if r["label_source"] == source]
    return [r for s in ("train", "test") for r in random.sample([x for x in pool if split[x["name"]] == s], per_split)]


generated = stratified("generated_from_ocr_text", 20)
truth = stratified("ground_truth", 10)

with open("label_audit/audit_sample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "split", "label_source", "needs_review", "audit_group"])
    for group, items in (("needs_review", review), ("generated", generated), ("ground_truth", truth)):
        for r in items:
            w.writerow([r["name"], split[r["name"]], r["label_source"], r["needs_review"], group])
print(len(review), len(generated), len(truth))
