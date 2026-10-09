#!/usr/bin/env python3
"""
build_teamdna_data.py — Team style-profile index for the Team DNA demo page.
Aggregates merged_index.json's per-player radar (att/cre/tec/def/tac
percentiles) into a minutes-weighted team average, split by team+league+season
so the same team can be compared across two different seasons.
"""
import json
import os
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "merged_index.json")
OUT = os.path.join(HERE, "data", "teamdna_index.json")

AXES = ["att", "cre", "tec", "def", "tac"]
MIN_MINS = 180


def main():
    players = json.load(open(SRC, encoding="utf-8"))
    groups = collections.defaultdict(list)
    for p in players:
        radar = p.get("radar")
        if not radar or not p.get("team") or not p.get("league") or not p.get("season"):
            continue
        if (p.get("mins") or 0) < MIN_MINS:
            continue
        if not any(radar.get(a) for a in AXES):
            continue
        key = (p["team"], p["league"], p["season"])
        groups[key].append(p)

    rows = []
    for (team, league, season), members in groups.items():
        if len(members) < 5:
            continue
        total_mins = sum(p["mins"] for p in members)
        radar = {}
        for ax in AXES:
            weighted = sum((p["radar"].get(ax) or 0) * p["mins"] for p in members)
            radar[ax] = round(weighted / total_mins, 1)
        rows.append({
            "team": team,
            "league": league,
            "season": season,
            "n_players": len(members),
            "mins_total": total_mins,
            "avg_age": round(sum((p.get("age") or 0) for p in members) / len(members), 1),
            "radar": radar,
        })

    rows.sort(key=lambda r: (r["league"], r["team"], r["season"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{len(rows)} team-seasons written -> {OUT}")


if __name__ == "__main__":
    main()
