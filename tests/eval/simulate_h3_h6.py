"""Hypothesis tests H3 and H6 for the Legal GraphRAG project.

H3 (verifier):  H0 the verifier does not change the hallucinated-citation rate.
                H1 it lowers it.   Paired binary per answer, before vs after.  Exact McNemar test.
H6 (latency):   H0 proposed and baseline have the same latency.
                H1 they differ.    Paired continuous per query.  Paired t-test on log latency,
                with Wilcoxon signed-rank as a check, plus the p95 <= 20 s criterion from the roadmap.

RUN MODES
  python simulate_h3_h6.py                       # SIMULATED data (placeholders, see ASSUMPTION lines)
  python simulate_h3_h6.py --h3-csv h3.csv --h6-csv h6.csv   # REAL data, same analysis
  h3.csv columns: before,after       (0/1 per answer; 1 = at least one cited id was not retrieved)
  h6.csv columns: base_s,prop_s      (seconds per query, same query in the same row)

Simulated output is NOT a result. The effect sizes below are inputs chosen by the author,
so any p-value computed from them only demonstrates that the analysis code works.
"""
import argparse
import csv
import math
import pathlib

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ ASSUMPTIONS (replace with measured values)
N_ANSWERS = 60                 # ASSUMPTION: size of the evaluation set (roadmap asks for at least 30)
P_CITATION_UNRETRIEVED = 0.06  # ASSUMPTION: chance one cited id is not in the retrieved set, per citation
MEAN_CITATIONS = 3.0           # ASSUMPTION: average provisions cited per answer (at least 1)
# seconds, (median, sigma of the log). ASSUMPTION: guesses to be replaced by timings measured on your laptop
COMP = {
    "embed":        (0.08, 0.20),
    "search":       (0.15, 0.25),
    "rerank":       (0.25, 0.20),
    "expand":       (0.10, 0.25),
    "verify":       (0.05, 0.20),
    "llm_plan":     (0.80, 0.35),
    "llm_issue":    (0.90, 0.35),
    "llm_answer_b": (1.50, 0.35),   # baseline answer (shorter context)
    "llm_answer_p": (2.00, 0.35),   # proposed answer (case + legal + graph context)
}
LOAD_SIGMA = 0.15              # ASSUMPTION: per-query common slowness (long question, busy API); creates pairing
P95_LIMIT_S = 20.0             # from docs/05-roadmap.md, Phase 4 exit criteria
ALPHA = 0.05


def draw(rng, key, n):
    med, sig = COMP[key]
    return rng.lognormal(math.log(med), sig, n)


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    # H3: each answer cites k provisions; flagged if at least one cited id was not retrieved.
    k = 1 + rng.poisson(MEAN_CITATIONS - 1, n)
    before = (rng.random(n) < 1 - (1 - P_CITATION_UNRETRIEVED) ** k).astype(int)
    after = np.zeros(n, dtype=int)  # R-21: the verifier removes every citation not in the retrieved set
    # H6: latency = sum of pipeline components, scaled by a per-query load factor shared by both systems.
    load = rng.lognormal(0.0, LOAD_SIGMA, n)
    base = load * (draw(rng, "embed", n) + draw(rng, "search", n) + draw(rng, "rerank", n) + draw(rng, "llm_answer_b", n))
    prop = load * (draw(rng, "llm_plan", n) + draw(rng, "embed", n) + draw(rng, "search", n) + draw(rng, "rerank", n)
                   + draw(rng, "llm_issue", n) + draw(rng, "search", n) + draw(rng, "expand", n) + draw(rng, "rerank", n)
                   + draw(rng, "llm_answer_p", n) + draw(rng, "verify", n))
    return before, after, base, prop


def analyze_h3(before, after):
    before, after = np.asarray(before, int), np.asarray(after, int)
    n = len(before)
    b = int(((before == 1) & (after == 0)).sum())   # flagged before, clean after (fixed)
    c = int(((before == 0) & (after == 1)).sum())   # clean before, flagged after (broken)
    out = {"n": n, "fixed_b": b, "broken_c": c,
           "rate_before": before.mean(), "rate_after": after.mean()}
    if b + c == 0:
        out["p_two_sided"] = out["p_one_sided"] = 1.0
    else:
        out["p_two_sided"] = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        out["p_one_sided"] = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue  # H1: lowers
    out["ci_before"] = stats.binomtest(int(before.sum()), n).proportion_ci(method="wilson")
    out["ci_after"] = stats.binomtest(int(after.sum()), n).proportion_ci(method="exact")  # Clopper-Pearson
    return out


def analyze_h6(base, prop, n_boot=10_000, seed=0):
    base, prop = np.asarray(base, float), np.asarray(prop, float)
    n = len(base)
    d = np.log(prop) - np.log(base)                  # per-query log ratio
    t = stats.ttest_1samp(d, 0.0)
    se = d.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    ratio = math.exp(d.mean())
    ratio_ci = (math.exp(d.mean() - tcrit * se), math.exp(d.mean() + tcrit * se))
    w = stats.wilcoxon(prop - base)
    rng = np.random.default_rng(seed)
    p95_boot = [np.percentile(rng.choice(prop, n, replace=True), 95) for _ in range(n_boot)]
    return {"n": n, "median_base": float(np.median(base)), "median_prop": float(np.median(prop)),
            "p95_base": float(np.percentile(base, 95)), "p95_prop": float(np.percentile(prop, 95)),
            "p95_prop_ci": (float(np.percentile(p95_boot, 2.5)), float(np.percentile(p95_boot, 97.5))),
            "ratio": ratio, "ratio_ci": ratio_ci, "t": float(t.statistic), "df": n - 1,
            "p_t": float(t.pvalue), "p_wilcoxon": float(w.pvalue),
            "cohen_dz": float(d.mean() / d.std(ddof=1))}


def load_csv(path, cols):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [np.array([float(r[c]) for r in rows]) for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3-csv")
    ap.add_argument("--h6-csv")
    ap.add_argument("--n", type=int, default=N_ANSWERS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=".", help="folder for the simulated CSVs")
    a = ap.parse_args()

    before, after, base, prop = simulate(a.n, a.seed)
    sim_h3, sim_h6 = not a.h3_csv, not a.h6_csv
    if a.h3_csv:
        before, after = load_csv(a.h3_csv, ["before", "after"])
    if a.h6_csv:
        base, prop = load_csv(a.h6_csv, ["base_s", "prop_s"])
    if sim_h3 or sim_h6:
        out = pathlib.Path(a.out)
        if sim_h3:
            with open(out / "SIMULATED_h3.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["before", "after"]); w.writerows(zip(before.astype(int), after.astype(int)))
        if sim_h6:
            with open(out / "SIMULATED_h6.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["base_s", "prop_s"]); w.writerows(zip(np.round(base, 4), np.round(prop, 4)))
        print("*** SIMULATED DATA: placeholders, not measurements. Do not report as results. ***\n")

    h3 = analyze_h3(before, after)
    print("H3  verifier lowers hallucinated-citation rate")
    print(f"  answers n = {h3['n']}")
    print(f"  rate before = {h3['rate_before']:.3f}  95% Wilson CI [{h3['ci_before'].low:.3f}, {h3['ci_before'].high:.3f}]")
    print(f"  rate after  = {h3['rate_after']:.3f}  95% exact CI  [{h3['ci_after'].low:.3f}, {h3['ci_after'].high:.3f}]")
    print(f"  discordant pairs: fixed b = {h3['fixed_b']}, broken c = {h3['broken_c']}")
    print(f"  exact McNemar  p (two-sided) = {h3['p_two_sided']:.4g}   p (one-sided, H1 lowers) = {h3['p_one_sided']:.4g}")

    h6 = analyze_h6(base, prop)
    print("\nH6  latency, proposed vs baseline")
    print(f"  queries n = {h6['n']}")
    print(f"  median  base = {h6['median_base']:.2f} s   proposed = {h6['median_prop']:.2f} s")
    print(f"  p95     base = {h6['p95_base']:.2f} s   proposed = {h6['p95_prop']:.2f} s  (95% bootstrap CI {h6['p95_prop_ci'][0]:.2f} to {h6['p95_prop_ci'][1]:.2f})")
    print(f"  geometric mean ratio proposed/base = {h6['ratio']:.2f}  95% CI [{h6['ratio_ci'][0]:.2f}, {h6['ratio_ci'][1]:.2f}]")
    print(f"  paired t on log latency: t({h6['df']}) = {h6['t']:.2f}, p = {h6['p_t']:.3g};  Wilcoxon p = {h6['p_wilcoxon']:.3g};  Cohen dz = {h6['cohen_dz']:.2f}")
    verdict = "MET" if h6["p95_prop_ci"][1] <= P95_LIMIT_S else "NOT shown to be met"
    print(f"  roadmap criterion p95 <= {P95_LIMIT_S:.0f} s (upper CI bound): {verdict}")


if __name__ == "__main__":
    main()
