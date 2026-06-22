#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

analyze_metrics.py  (nadogradjeno)

==================================

Agregacija metrika iz metrics_*.csv (baseline / shield / shieldnogap).



NADOGRADNJE u ovoj verziji:

  * Automatski bira rutu po scenariju: S1 -> route_intersection_test_1.xml,

    svi ostali -> route_spawn0.xml. Vise ne treba rucno razdvajati S1.

  * Prepoznaje mod 'shieldnogap' (ablacija) uz 'shield' i 'baseline'.

  * Kanonski grupira scenarije (s1..s7) bez obzira na tocan oblik imena;

    S7 grupira po brzini (s7_v4, s7_v8, ...) pa se modovi poklope.

  * 'ruta_zavrsena' prikazuje se samo za S1-S4 (gdje je to mjera).

    Za S5-S7 stoji '-' (te su voznje prekidane; citaj stupac 'sudari').



POKRETANJE:

    python analyze_metrics.py                      # trazi u ./tests, pa ./ ; auto-rute

    python analyze_metrics.py --dir tests          # eksplicitno zadaj folder s CSV-ovima

    python analyze_metrics.py "metrics_*s7*.csv"   # samo uzorak

    python analyze_metrics.py --route ruta.xml --route-s1 ruta_s1.xml metrics_*.csv



Rezultat: tablica u konzoli + summary.csv (po voznji) + summary_grouped.csv.

"""



import csv

import sys

import os

import glob

import re

import argparse



try:

    import xml.etree.ElementTree as ET

except Exception:

    ET = None





# scenariji koji se ocjenjuju dovrsenoscu rute (ostali: sudari/zaustavljanje)

ROUTE_SCENARIOS = {"s1", "s2", "s3", "s4"}

# scenariji koji koriste S1 rutu (ostali koriste route_spawn0)

S1_SCENARIOS = {"s1"}





# ---------------------------------------------------------------- pomocno

def load_goal(route_xml):

    """Vrati (x, y) zadnjeg waypointa rute = cilj."""

    if ET is None or not route_xml or not os.path.exists(route_xml):

        return None

    wps = ET.parse(route_xml).findall(".//waypoint")

    if not wps:

        return None

    last = wps[-1]

    return float(last.get("x")), float(last.get("y"))



def resolve(name, extra_dirs):

    """Nadji datoteku: prvo kako je zadano, pa po imenu u extra_dirs."""

    cands = [name] + [os.path.join(d, os.path.basename(name)) for d in extra_dirs]

    for c in cands:

        if c and os.path.exists(c):

            return c

    return name





def classify(path):

    """Vrati (scenarij_label, kategorija). kategorija in s1..s7 ili None.

    Za S7 label ukljucuje brzinu (npr. 's7_v4') da se modovi poklope po brzini."""

    b = os.path.splitext(os.path.basename(path))[0].lower()

    if "s1_raskrizje" in b:

        return "s1", "s1"

    if "s2_global" in b:

        return "s2", "s2"

    if "s3_10v_7w" in b:

        return "s3", "s3"

    if "20v_11w" in b:

        return "s4", "s4"

    if "_s5_" in b:

        return "s5", "s5"

    if "_s6_" in b:

        return "s6", "s6"

    if "s7" in b:

        m = re.search(r"v(\d+)", b)

        return (f"s7_v{m.group(1)}" if m else "s7"), "s7"

    return None, None





def parse_name(path):

    """Fallback iz imena 'metrics_<mode>_<scenarij>_<rep>.csv' -> (scenarij, rep)."""

    base = os.path.splitext(os.path.basename(path))[0]

    parts = base.split("_")

    if len(parts) >= 4 and parts[0] == "metrics" and parts[1] in ("shield", "baseline", "shieldnogap"):

        rep = parts[-1]

        scenarij = "_".join(parts[2:-1])

        try:

            rep = int(rep)

        except ValueError:

            scenarij = "_".join(parts[2:])

            rep = 1

        return scenarij or base, rep

    return base, 1





def detect_rep(path):

    """Ponavljanje = zadnji broj u imenu (cosmeticno; grupiranje broji retke)."""

    base = os.path.splitext(os.path.basename(path))[0]

    m = re.search(r"(\d+)$", base)

    return int(m.group(1)) if m else 1





# ---------------------------------------------------------------- analiza 1 CSV

def analyze(path, goal=None, scen_label=None, category=None):

    with open(path, encoding="utf-8") as f:

        rows = list(csv.DictReader(f))

    if not rows:

        return None



    n = len(rows)

    lat = [abs(float(r["lateral_error"])) for r in rows]

    spd = [float(r["speed"]) for r in rows]

    coll = int(rows[-1]["collision_count"])

    yld = sum(1 for r in rows if r["intersection_risk"] == "YIELD")

    stp = sum(1 for r in rows if r["intersection_risk"] == "STOP")

    offlane = sum(1 for v in lat if v > 1.0)

    lane_rc = sum(1 for r in rows if r["lane_recenter"] == "True")

    steer_diff = sum(1 for r in rows

                     if abs(float(r["steer"]) - float(r["raw_steer"])) > 0.001)

    ex, ey = float(rows[-1]["x"]), float(rows[-1]["y"])



    if scen_label is None:

        scen_label, rep = parse_name(path)

        category = scen_label

    else:

        rep = detect_rep(path)



    res = {

        "file": os.path.basename(path),

        "scenarij": scen_label,

        "kategorija": category or scen_label,

        "mode": rows[0]["mode"],

        "rep": rep,

        "koraka": n,

        "sudari": coll,

        "max_lateral": round(max(lat), 3),

        "avg_lateral": round(sum(lat) / n, 3),

        "izvan_trake": offlane,

        "yield": yld,

        "stop": stp,

        "lane_recenter": lane_rc,

        "avg_speed": round(sum(spd) / n, 2),

        "kraj_x": round(ex, 1),

        "kraj_y": round(ey, 1),

        "cisti_baseline": (steer_diff == 0),

    }

    is_route = (res["kategorija"] in ROUTE_SCENARIOS)

    if goal is not None:

        d = ((ex - goal[0]) ** 2 + (ey - goal[1]) ** 2) ** 0.5

        res["dist_do_cilja"] = round(d, 1)

        res["ruta_zavrsena"] = (d < 6.0) if is_route else "-"

    else:

        res["dist_do_cilja"] = ""

        res["ruta_zavrsena"] = ("?" if is_route else "-")

    return res





# ---------------------------------------------------------------- ispis tablice

def print_table(rows, cols):

    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    line = "  ".join(str(c).ljust(widths[c]) for c in cols)

    print(line)

    print("-" * len(line))

    for r in rows:

        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))





def write_csv(path, rows, cols):

    with open(path, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=cols)

        w.writeheader()

        for r in rows:

            w.writerow({c: r.get(c, "") for c in cols})





# ---------------------------------------------------------------- grupiranje

def group(results, have_goal):

    """Grupiraj po (scenarij, mode) i usredni numericke metrike po ponavljanjima."""

    groups = {}

    for r in results:

        groups.setdefault((r["scenarij"], r["mode"]), []).append(r)



    num = ["koraka", "sudari", "max_lateral", "avg_lateral", "izvan_trake",

           "yield", "stop", "lane_recenter", "avg_speed"]

    out = []

    for (scen, mode), rs in sorted(groups.items()):

        g = {"scenarij": scen, "mode": mode, "n_voznji": len(rs)}

        for k in num:

            g[k] = round(sum(x[k] for x in rs) / len(rs), 3)

        cat = rs[0]["kategorija"]

        if have_goal:

            if cat in ROUTE_SCENARIOS:

                done = sum(1 for x in rs if x.get("ruta_zavrsena") is True)

                g["ruta_zavrsena"] = f"{done}/{len(rs)}"

            else:

                g["ruta_zavrsena"] = "-"

        if mode == "baseline":

            g["cisti_baseline"] = all(x["cisti_baseline"] for x in rs)

        out.append(g)

    return out





# ---------------------------------------------------------------- main

def main():

    ap = argparse.ArgumentParser(description="Agregacija metrika baseline vs shield vs shieldnogap.")

    ap.add_argument("files", nargs="*", help="CSV fajlovi ili uzorci (default: metrics_*.csv)")

    ap.add_argument("--route", default="route_spawn0.xml",

                    help="route XML za S2-S7 (default: route_spawn0.xml)")

    ap.add_argument("--route-s1", dest="route_s1", default="route_intersection_test_1.xml",

                    help="route XML za S1 (default: route_intersection_test_1.xml)")

    ap.add_argument("--dir", default="tests",

                    help="folder u kojem su metrics_*.csv (default: tests; pada na ./)")

    ap.add_argument("--out", default="summary.csv", help="izlazni CSV po voznji")

    args = ap.parse_args()



    if args.files:

        patterns = args.files

    else:

        patterns = [os.path.join(args.dir, "metrics_*.csv"), "metrics_*.csv"]

    paths = []

    for p in patterns:

        paths.extend(glob.glob(p))

    paths = sorted(set(p for p in paths if os.path.basename(p) not in

                       (args.out, "summary_grouped.csv")))

    if not paths:

        print(f"Nema CSV fajlova u '{args.dir}' ni u trenutnom folderu.")

        print("Primjer: python analyze_metrics.py --dir tests")

        sys.exit(1)



    _csv_dir = os.path.dirname(paths[0]) if paths else "."

    _search = [args.dir, ".", _csv_dir]

    goal_spawn0 = load_goal(resolve(args.route, _search))

    goal_s1 = load_goal(resolve(args.route_s1, _search))

    if goal_spawn0 is None:

        print(f"[upozorenje] ne mogu procitati {args.route} - ruta_zavrsena za S2-S4 ce biti '?'.")

    if goal_s1 is None:

        print(f"[upozorenje] ne mogu procitati {args.route_s1} - ruta_zavrsena za S1 ce biti '?'.")

    have_goal = (goal_spawn0 is not None) or (goal_s1 is not None)



    results = []

    for p in paths:

        try:

            lab, cat = classify(p)

            goal = goal_s1 if (cat in S1_SCENARIOS) else goal_spawn0

            r = analyze(p, goal, lab, cat)

            if r:

                results.append(r)

        except Exception as e:

            print(f"[greska] {p}: {e}")



    if not results:

        print("Nista za analizu.")

        sys.exit(1)



    per_cols = ["file", "scenarij", "mode", "rep", "koraka", "sudari",

                "max_lateral", "avg_lateral", "izvan_trake", "yield", "stop",

                "lane_recenter", "avg_speed", "kraj_x", "kraj_y", "cisti_baseline"]

    if have_goal:

        per_cols += ["dist_do_cilja", "ruta_zavrsena"]



    print("\n=== PO VOZNJI ===")

    print_table(results, per_cols)

    write_csv(args.out, results, per_cols)



    grp = group(results, have_goal)

    grp_cols = ["scenarij", "mode", "n_voznji", "koraka", "sudari",

                "max_lateral", "avg_lateral", "izvan_trake", "yield", "stop",

                "lane_recenter", "avg_speed"]

    if have_goal:

        grp_cols.append("ruta_zavrsena")

    grp_cols.append("cisti_baseline")



    print("\n=== GRUPIRANO (prosjek po ponavljanjima) ===")

    print_table(grp, grp_cols)

    write_csv("summary_grouped.csv", grp, grp_cols)



    bad = [r for r in results if r["mode"] == "baseline" and not r["cisti_baseline"]]

    if bad:

        print("\n[UPOZORENJE] baseline voznje gdje steer != raw_steer (NIJE cisti baseline):")

        for r in bad:

            print("   ", r["file"])



    print(f"\nSpremljeno: {args.out} (po voznji), summary_grouped.csv (grupirano).")





if __name__ == "__main__":

    main()