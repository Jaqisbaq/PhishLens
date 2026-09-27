"""Part A step 2: group by registered domain, stratified group split.

Development 70 percent, locked test 30 percent, fixed seed 2026. No domain
appears in both partitions. Writes a split manifest with a sha256 hash.
"""
from __future__ import annotations

import hashlib
import json
import random

import pandas as pd

import common

CASES = common.CACHE_DIR / "case_table.parquet"
OUT_SPLIT = common.CACHE_DIR / "split.parquet"
OUT_MANIFEST = common.OUTPUTS_DIR / "split_manifest.json"


def main() -> None:
    cases = pd.read_parquet(CASES)
    rng = random.Random(common.SEED)

    # Per domain: label counts (a domain could in principle mix labels; here
    # each domain is assigned by its majority label so the whole domain group
    # stays on one side of the split).
    dom_labels = cases.groupby("domain")["label"].agg(lambda s: s.mode().iloc[0])
    dom_counts = cases.groupby("domain")["label"].count()

    domains_phish = [d for d, l in dom_labels.items() if l == 1]
    domains_legit = [d for d, l in dom_labels.items() if l == 0]
    rng.shuffle(domains_phish)
    rng.shuffle(domains_legit)

    def greedy_split(domains, target_test_frac):
        total = sum(dom_counts[d] for d in domains)
        target_test = total * target_test_frac
        test, dev = [], []
        running = 0
        for d in domains:
            if running < target_test:
                test.append(d)
                running += dom_counts[d]
            else:
                dev.append(d)
        return dev, test

    dev_p, test_p = greedy_split(domains_phish, 0.30)
    dev_l, test_l = greedy_split(domains_legit, 0.30)

    dev_domains = set(dev_p) | set(dev_l)
    test_domains = set(test_p) | set(test_l)
    assert not (dev_domains & test_domains)

    cases = cases.copy()
    cases["split"] = cases["domain"].map(lambda d: "test" if d in test_domains else "dev")

    cases[["id", "domain", "label", "split"]].to_parquet(OUT_SPLIT, index=False)

    dev = cases[cases.split == "dev"]
    test = cases[cases.split == "test"]

    manifest = {
        "seed": common.SEED,
        "grouping": "registered domain of the case URL (string field 'domain' from the "
                    "linked-check log, no network lookup)",
        "method": "domains shuffled with the fixed seed, then greedily assigned so each "
                  "class reaches approximately 30 percent of its cases in the test "
                  "partition, domain-disjoint",
        "n_domains_total": int(cases["domain"].nunique()),
        "n_domains_dev": len(dev_domains),
        "n_domains_test": len(test_domains),
        "dev": {"total": int(len(dev)), "phishing": int((dev.label == 1).sum()),
                "legitimate": int((dev.label == 0).sum())},
        "test": {"total": int(len(test)), "phishing": int((test.label == 1).sum()),
                 "legitimate": int((test.label == 0).sum())},
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    manifest["sha256_of_manifest_body"] = sha
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps(manifest, indent=2))
    print("wrote", OUT_SPLIT)
    print("wrote", OUT_MANIFEST)


if __name__ == "__main__":
    main()
