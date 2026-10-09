#!/usr/bin/env python3
"""
build_breakout_data.py — Season-over-season delta index for the Breakout
Tracker demo page. Groups fotmob_index.json entries by base_id, keeps
players with two valid season entries (rating present, mp>=5, cats present),
takes the two most recent by season year, and computes per-match goal/assist
rates plus a rating delta and a composite breakout score.
"""
import json
import os
import re
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "fotmob_index.json")
OUT = os.path.join(HERE, "data", "breakout_index.json")

MIN_MP = 5


def season_year(s):
    m = re.match(r"(\d{4})", s or "")
    return int(m.group(1)) if m else 0


def goals_assists(p):
    att = (p.get("cats") or {}).get("Attacking") or {}
    g = att.get("goals")
    a = att.get("assists")
    return (g[0] if g else 0.0), (a[0] if a else 0.0)


def main():
    players = json.load(open(SRC, encoding="utf-8"))
    groups = collections.defaultdict(list)
    for p in players:
        groups[p["base_id"]].append(p)

    rows = []
    for base_id, entries in groups.items():
        valid = [
            p for p in entries
            if p.get("rating") is not None
            and (p.get("mp") or 0) >= MIN_MP
            and (p.get("cats") or {}).get("Attacking")
        ]
        if len(valid) < 2:
            continue
        valid.sort(key=lambda p: season_year(p["season"]), reverse=True)
        cur, prev = valid[0], valid[1]
        if season_year(cur["season"]) == season_year(prev["season"]):
            continue

        g_cur, a_cur = goals_assists(cur)
        g_prev, a_prev = goals_assists(prev)
        gpm_cur, gpm_prev = g_cur / cur["mp"], g_prev / prev["mp"]
        apm_cur, apm_prev = a_cur / cur["mp"], a_prev / prev["mp"]
        rating_delta = round(cur["rating"] - prev["rating"], 2)
        ga_pm_delta = (gpm_cur + apm_cur) - (gpm_prev + apm_prev)
        score = round(rating_delta * 10 + ga_pm_delta * 20, 2)

        rows.append({
            "id": cur["id"],
            "base_id": base_id,
            "name": cur["name"],
            "team": cur.get("team"),
            "league": cur.get("league"),
            "pos": cur.get("pos"),
            "pos_key": cur.get("pos_key"),
            "age": cur.get("age"),
            "mv": cur.get("mv"),
            "photo": cur.get("photo"),
            "season_cur": cur["season"],
            "season_prev": prev["season"],
            "mp_cur": cur["mp"],
            "mp_prev": prev["mp"],
            "mins_cur": cur.get("mins"),
            "mins_prev": prev.get("mins"),
            "rating_cur": cur["rating"],
            "rating_prev": prev["rating"],
            "rating_delta": rating_delta,
            "goals_cur": round(g_cur, 1),
            "goals_prev": round(g_prev, 1),
            "assists_cur": round(a_cur, 1),
            "assists_prev": round(a_prev, 1),
            "gpm_cur": round(gpm_cur, 2),
            "gpm_prev": round(gpm_prev, 2),
            "apm_cur": round(apm_cur, 2),
            "apm_prev": round(apm_prev, 2),
            "score": score,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{len(rows)} players written -> {OUT}")


if __name__ == "__main__":
    main()
