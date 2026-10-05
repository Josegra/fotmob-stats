#!/usr/bin/env python3
"""
build_fotmob_data.py — Builds per-player JSON for the Fotmob player profile page
(fotmob_player.html) from newgen_stats/fotmob_v2_profiles.csv.

Outputs:
    data/fotmob/{player_id}.json   Bio + grouped stats + heatmap + shot map
    data/fotmob_index.json         Lightweight list for search (id, name, team, league, pos, mv)

Market value time series is NOT duplicated here — fotmob_player.html reads the
existing data/valuations/{player_id}.json directly (same player_id key space,
already built by build_data.py from datasources_football/valuations.csv).

Run from webapp/ folder:
    python build_fotmob_data.py
"""
import json, math, re, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

# scrape_fotmob_v2.py has a bug (confirmed 2026-10-02, 653 names / 113 legacy
# 'team' values affected): some non-ASCII names get saved with a literal
# "é"-style escape instead of the real character. Un-escape it here so the
# page displays correctly regardless of whether the source CSV gets fixed.
_UESC_RE = re.compile(r'\\u([0-9a-fA-F]{4})')
def unescape_literal_unicode(s):
    if not isinstance(s, str) or '\\u' not in s:
        return s
    return _UESC_RE.sub(lambda m: chr(int(m.group(1), 16)), s)

WEBAPP   = Path(__file__).parent
ROOT     = WEBAPP.parent
SRC_CSV  = ROOT / 'newgen_stats' / 'fotmob_v2_profiles.csv'
PREV_SEASON_CSV = ROOT / 'newgen_stats' / 'fotmob_v2_profiles_PREV_SEASON.csv'
VAL_CSV  = ROOT / 'datasources_football' / 'valuations.csv'
OUT_DIR  = WEBAPP / 'data' / 'fotmob'
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR = WEBAPP / 'data' / 'valuations'
VAL_DIR.mkdir(parents=True, exist_ok=True)

# fotmob_v2_profiles_PREV_SEASON.csv mixes two different things under one
# filename: for split-calendar leagues (Premier League, La Liga, ...) it's a
# genuine prior season ("25/26" vs the main CSV's current "2026-2027"). For
# calendar-year leagues (MLS, Brazil, Chile, Ecuador) it's actually just an
# EARLIER CHECKPOINT of the SAME current season — confirmed 2026-10-04 by
# comparing matches_played against the main CSV for overlapping player_ids
# (e.g. an MLS player showed 18 games in this file vs 25 in the main CSV,
# same season). Only the leagues below hold real, distinct prior-season data;
# the rest are excluded so the Season filter never shows a fake duplicate.
PREV_SEASON_EXCLUDE_LEAGUES = {'Brazil Serie A', 'Chile Primera Division', 'Ecuador Liga Pro', 'MLS'}
PREV_SEASON_LABEL = '2025-2026'

# Individual PREV_SEASON rows confirmed bad on a case-by-case basis — unlike
# STALE_CURRENT_SEASON_LEAGUES below, there's no reliable data-only signal for
# this one. Checked 2026-10-06: comparing matches_played for international vs
# non-international players in PREV_SEASON turned up no distinguishing pattern
# (54% vs 57% with <=10 matches) — so a blanket "international players are
# suspect" rule would delete plenty of genuine partial seasons too. Fotmob's
# API returns whatever is contextually "current" at scrape time regardless of
# the tournamentId requested (confirmed live: querying World Cup tournamentId
# 77 and LaLiga tournamentId 87 for the same player returned identical data),
# so a row scraped during an international window can end up with national-team
# form under a club league's label. Add player_ids here only once confirmed —
# e.g. via live playerData comparison like the Lamine/Nelsson checks — not on
# a hunch, since the risk of wrongly deleting a real injury-shortened season is
# real. Lamine Yamal (1467236): user-flagged 2026-10-06, his PREV_SEASON row
# (8 matches/1 goal/615 min) doesn't match a live fetch of either his current
# club form or World Cup tournament stats, but the pattern (short, high-
# intensity appearance burst) is consistent enough with the international-
# contamination theory that the user asked for it to be removed outright.
PREV_SEASON_EXCLUDE_PLAYER_IDS = {1467236}

# Pattern-based exclusion (2026-10-06): can't live-verify historical data (no
# time machine — a live fetch only ever shows CURRENT state, which is what
# confirmed Nelsson/Hull City but can't confirm a PAST season's row), so this
# is evidence-by-pattern, not proof, and was a deliberate tradeoff the user
# accepted. At market_value>=30M + is_international + matches_played<=8, the
# candidate list (166 players in the 14 euro leagues) is almost entirely
# undisputed starters who would routinely play 30-40 games a season absent a
# major injury — Haaland, Bellingham, Mbappe, Vinicius Jr, Pedri, Saka,
# Musiala and Kane were all in it. At that concentration of elite, rarely-
# rotated players, a single-digit "full season" match count is far more likely
# to be the same tournamentId-ignoring contamination as Lamine's case than
# 166 simultaneous career-threatening injuries. Residual risk: a genuine
# long-term injury (e.g. an ACL tear) looks identical in the data and could be
# wrongly swept up here — accepted knowingly, not a false positive we can
# detect and exclude.
PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MP = 8
PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MV = 30_000_000

# Split-calendar European leagues (season runs ~August to May). A "current
# 2026-2027" row showing an implausibly high matches_played this early in the
# season means the scraper never refreshed that row after the season rolled
# over — it's really the player's complete FINAL 2025/26 tally, mislabeled as
# current. Confirmed live 2026-10-05: Victor Nelsson's "current" row showed 38
# Serie A matches (a league/club he'd actually left for Superligaen, where his
# real 2026-27 total was 3) — and the matches_played distribution for these
# leagues has a clean valley between 9 and 15, not a smooth curve, confirming
# this is a distinct stale-data population rather than genuinely busy players.
# Every one of the ~809 affected players already has a real 2025-2026 row, so
# dropping the bad "current" row never removes a player from the site outright
# — it just stops showing them under a season they don't have real data for.
STALE_CURRENT_SEASON_LEAGUES = {
    'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Championship',
    'La Liga 2', '2. Bundesliga', 'Eredivisie', 'Primeira Liga', 'Swiss Super League',
    'Russian Premier League', 'Saudi Pro League', 'Super Lig',
}
STALE_CURRENT_SEASON_MP_THRESHOLD = 14


def build_valuation_files():
    """webapp/data/valuations/{player_id}.json used to be written only by
    build_data.py, gated behind ITS OWN name-matching into the old multi-source
    players.json — a player missing from that fuzzy match (confirmed: Lamine
    Yamal, who has 44 real rows in valuations.csv) silently got no file at all.
    valuations.csv is already keyed by this exact Fotmob player_id, so write it
    straight from there — no name-matching needed, no dependency on build_data.py.

    Also returns a {player_id: [v0..v7]} trend map — up to 8 evenly-sampled
    points from each player's value history — so the leaderboard can draw an
    inline sparkline per row straight from fotmob_index.json, no per-row fetch.
    """
    if not VAL_CSV.exists():
        print(f'  [warn] {VAL_CSV} not found — skipping valuation history files')
        return 0, {}
    vdf = pd.read_csv(VAL_CSV, parse_dates=['date'])
    vdf = vdf.dropna(subset=['player_id']).sort_values('date')
    written = 0
    trend_map = {}
    for pid, grp in vdf.groupby('player_id'):
        pid = int(pid)
        values = [int(v) if pd.notna(v) else None for v in grp['value_eur']]
        data = {
            'dates':  grp['date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': values,
            'lower':  [int(v) if pd.notna(v) else None for v in grp.get('lower_eur', [])],
            'upper':  [int(v) if pd.notna(v) else None for v in grp.get('upper_eur', [])],
        }
        with open(VAL_DIR / f'{pid}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
        written += 1
        clean = [v for v in values if v is not None]
        if len(clean) >= 2:
            if len(clean) <= 8:
                trend_map[pid] = clean
            else:
                idx = np.linspace(0, len(clean) - 1, 8).round().astype(int)
                trend_map[pid] = [clean[i] for i in idx]
    return written, trend_map

# ── Stat groups: label -> list of (key, display label, higher_is_better) ──────
# key matches the CSV's {key}_val / {key}_per90 / {key}_pct / {key}_pct90 columns.
GROUPS_OUTFIELD = {
    'Attacking': [
        ('goals', 'Goals', 1), ('xg', 'xG', 1), ('xg_excl_penalty', 'xG (non-pen)', 1),
        ('assists', 'Assists', 1), ('xa', 'xA', 1),
        ('shots', 'Shots', 1), ('shots_on_target', 'Shots on Target', 1),
        ('big_chances_created', 'Big Chances Created', 1),
        ('big_chances_missed', 'Big Chances Missed', -1),
        ('penalty_goals', 'Penalty Goals', 1),
    ],
    'Passing & Creativity': [
        ('accurate_passes', 'Accurate Passes', 1), ('pass_accuracy', 'Pass Accuracy %', 1),
        ('accurate_long_balls', 'Accurate Long Balls', 1), ('long_ball_accuracy', 'Long Ball Accuracy %', 1),
        ('chances_created', 'Chances Created', 1),
        ('successful_crosses', 'Successful Crosses', 1), ('cross_accuracy', 'Cross Accuracy %', 1),
        ('line_breaking_passes', 'Line-Breaking Passes', 1),
    ],
    'Dribbling & Ball Carrying': [
        ('dribbles', 'Dribbles', 1), ('dribbles_success_rate', 'Dribble Success %', 1),
        ('touches', 'Touches', 1), ('touches_in_opposition_box', 'Touches in Box', 1),
        ('dispossessed', 'Dispossessed', -1), ('fouls_won', 'Fouls Won', 1),
    ],
    'Defending': [
        ('tackles', 'Tackles', 1), ('interceptions', 'Interceptions', 1),
        ('clearances', 'Clearances', 1), ('recoveries', 'Recoveries', 1),
        ('duels_won', 'Duels Won', 1), ('aerials_won', 'Aerials Won', 1),
        ('blocked_scoring_attempt', 'Shots Blocked', 1),
        ('dribbled_past', 'Dribbled Past', -1),
        ('possession_won_final_3rd', 'Poss. Won Final 3rd', 1),
        ('defensive_actions', 'Defensive Actions', 1),
        ('fouls_committed', 'Fouls Committed', -1),
    ],
    'Physical': [
        ('top_speed', 'Top Speed', 1), ('total_distance_covered', 'Distance Covered', 1),
        ('sprinting', 'Sprint Distance', 1), ('running', 'Running Distance', 1),
        ('number_of_sprints', 'Number of Sprints', 1),
    ],
    'Discipline': [
        ('yellow_cards', 'Yellow Cards', -1), ('red_cards', 'Red Cards', -1),
    ],
}

GROUPS_KEEPER = {
    'Goalkeeping': [
        ('saves', 'Saves', 1), ('save_percentage', 'Save %', 1),
        ('goals_conceded', 'Goals Conceded', -1), ('goals_prevented', 'Goals Prevented', 1),
        ('clean_sheets', 'Clean Sheets', 1), ('high_claims', 'High Claims', 1),
        ('penalty_saves', 'Penalty Saves', 1), ('acted_as_sweeper', 'Sweeper Actions', 1),
        ('error_led_to_goal', 'Errors Leading to Goal', -1),
    ],
    'Distribution': [
        ('accurate_passes', 'Accurate Passes', 1), ('pass_accuracy', 'Pass Accuracy %', 1),
        ('accurate_long_balls', 'Accurate Long Balls', 1), ('long_ball_accuracy', 'Long Ball Accuracy %', 1),
    ],
    'Physical': [
        ('top_speed', 'Top Speed', 1), ('total_distance_covered', 'Distance Covered', 1),
    ],
    'Discipline': [
        ('yellow_cards', 'Yellow Cards', -1), ('red_cards', 'Red Cards', -1),
    ],
}

# Union used by the leaderboard's category tabs: outfield categories as-is,
# plus Goalkeeping from GROUPS_KEEPER (its Distribution/Physical/Discipline
# are the same CSV columns as the outfield versions, already covered above).
LEADERBOARD_CATEGORIES = {**GROUPS_OUTFIELD, 'Goalkeeping': GROUPS_KEEPER['Goalkeeping']}

POSITION_LABELS = {
    'keeper_long': 'Goalkeeper', 'centerback': 'Centre-Back',
    'leftback': 'Left-Back', 'rightback': 'Right-Back',
    'left_wing_back': 'Left Wing-Back', 'right_wing_back': 'Right Wing-Back',
    'centerdefensivemidfielder': 'Defensive Midfield', 'centermidfielder': 'Central Midfield',
    'centerattackingmidfielder': 'Attacking Midfield',
    'leftmidfielder': 'Left Midfield', 'rightmidfielder': 'Right Midfield',
    'leftwinger': 'Left Winger', 'rightwinger': 'Right Winger',
    'striker': 'Striker',
}


def safe(v, nd=None):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), nd) if nd is not None else float(v)
    if isinstance(v, str):
        v = unescape_literal_unicode(v.strip())
        return v or None
    return v


def minutes_played_with_gk_estimate(row, is_gk, matches_played, matches_started):
    """Fotmob's API has no 'Minutes played' stat for goalkeepers at all (confirmed
    2026-10-05 against the live playerData endpoint — neither topStatCard nor
    statsSection carries it for the keeper position), so minutes_played is null
    for ~98% of keepers even when they started every match. A keeper who starts
    is essentially never substituted, so matches_started*90 (falling back to
    matches_played*90) is a safe, clearly-labelled estimate rather than leaving
    real starting keepers invisible to any minutes-based filter.
    """
    mins = safe(row.get('minutes_played'))
    if mins is not None or not is_gk:
        return mins
    basis = matches_started if matches_started is not None else matches_played
    return round(basis * 90) if basis is not None else None


def parse_json_col(v):
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if not s or s in ('nan', 'None', '[]'):
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def build_stats(row, groups):
    out = []
    for group_name, metrics in groups.items():
        rows = []
        for key, label, direction in metrics:
            val = safe(row.get(f'{key}_val'), 2)
            per90 = safe(row.get(f'{key}_per90'), 2)
            pct = safe(row.get(f'{key}_pct'))
            if val is None and per90 is None:
                continue
            rows.append({'key': key, 'label': label, 'val': val, 'per90': per90,
                         'pct': pct, 'dir': direction})
        if rows:
            out.append({'group': group_name, 'metrics': rows})
    return out


def build_shots(raw_shots):
    shots = parse_json_col(raw_shots)
    out = []
    for s in shots:
        if not isinstance(s, dict) or s.get('x') is None:
            continue
        outcome = s.get('outcome', '')
        out.append({
            'x': safe(s.get('x'), 1), 'y': safe(s.get('y'), 1),
            'xg': safe(s.get('xg'), 3), 'xgot': safe(s.get('xgot'), 3),
            'outcome': outcome, 'is_goal': outcome == 'Goal',
            'on_target': bool(s.get('on_target')), 'blocked': bool(s.get('blocked')),
            'foot': s.get('foot'), 'min': safe(s.get('min')),
            'situation': s.get('situation'), 'period': s.get('period'),
            # end-of-trajectory point (where it was blocked, saved or flew past the
            # frame) — Fotmob only fills this for most non-goal shots, null for goals
            'ex': safe(s.get('blocked_x'), 1), 'ey': safe(s.get('blocked_y'), 1),
        })
    return out


def build_heatmap_summary(row):
    # heatmap_{def,mid,att}_third and heatmap_{left,right}_half are RAW POINT
    # COUNTS per zone, not percentages (confirmed: def+mid+att == heatmap_points
    # exactly) — convert to shares of total here, once, so the page never has
    # to guess at the unit.
    total = safe(row.get('heatmap_points')) or 0
    def pct(key):
        v = safe(row.get(key))
        return round(v / total * 100, 1) if v is not None and total else None
    return {
        'points': total,
        'x_avg': safe(row.get('heatmap_x_avg'), 1),
        'y_avg': safe(row.get('heatmap_y_avg'), 1),
        'def_third': pct('heatmap_def_third'),
        'mid_third': pct('heatmap_mid_third'),
        'att_third': pct('heatmap_att_third'),
        'left_half': pct('heatmap_left_half'),
        'right_half': pct('heatmap_right_half'),
    }


def build_heatmap(raw_heat):
    pts = parse_json_col(raw_heat)
    out = []
    for p in pts:
        if isinstance(p, dict) and p.get('x') is not None and p.get('y') is not None:
            out.append([safe(p.get('x'), 1), safe(p.get('y'), 1)])
    return out


def has_valuation(player_id):
    return (VAL_DIR / f'{int(player_id)}.json').exists()


def process_dataframe(df, trend_map, season_index, out_id_fn, season_label_fn):
    """Writes data/fotmob/{out_id}.json for every row and returns the matching
    leaderboard-index entries. out_id_fn(pid)/season_label_fn(row) let the
    current-season pass keep today's plain-{pid}.json scheme untouched while
    the prior-season pass writes {pid}_{season}.json instead — same per-row
    logic either way, so both seasons stay in sync automatically.
    """
    index = []
    written = 0
    for _, row in df.iterrows():
        pid = row.get('player_id')
        if pd.isna(pid):
            continue
        pid = int(pid)
        out_id = out_id_fn(pid)
        season_label = season_label_fn(row)
        pos_key = row.get('position_key')
        is_gk = pos_key == 'keeper_long'
        groups = GROUPS_KEEPER if is_gk else GROUPS_OUTFIELD
        mp_val = safe(row.get('matches_played'))
        ms_val = safe(row.get('matches_started'))

        bio = {
            'id': out_id,
            'base_id': pid,
            'name': safe(row.get('name')),
            'team': safe(row.get('team_name')) or safe(row.get('team')),
            'team_id': safe(row.get('team_id')),
            'league': safe(row.get('league')),
            'position': POSITION_LABELS.get(pos_key, safe(row.get('position')) or ''),
            'position_key': safe(pos_key),
            'is_keeper': is_gk,
            'age': safe(row.get('bio_age')),
            'birth_date': safe(row.get('birth_date')),
            'height': safe(row.get('bio_height')),
            'foot': safe(row.get('bio_preferred_foot')),
            'country': safe(row.get('bio_country')),
            'shirt': safe(row.get('bio_shirt')),
            'contract_end': safe(row.get('bio_contract_end')),
            'market_value_eur': safe(row.get('market_value_eur')),
            'national_team': safe(row.get('national_team')),
            'is_international': bool(safe(row.get('is_international'))),
            'season': season_label,
            'matches_played': mp_val,
            'matches_started': ms_val,
            'minutes_played': minutes_played_with_gk_estimate(row, is_gk, mp_val, ms_val),
            'fotmob_rating': safe(row.get('fotmob_rating'), 2),
            'fotmob_rating_pct': safe(row.get('fotmob_rating_pct')),
            'photo_url': f'https://images.fotmob.com/image_resources/playerimages/{pid}.png',
            'has_valuation_history': has_valuation(pid),
        }

        player = {
            'bio': bio,
            'stats': build_stats(row, groups),
            'heatmap': build_heatmap(row.get('heatmap_coords')),
            'heatmap_summary': build_heatmap_summary(row),
            'shots': build_shots(row.get('shots_detail')),
            'shot_summary': {
                'total': safe(row.get('shots_total')),
                'on_target': safe(row.get('shots_on_target')),
                'blocked': safe(row.get('shots_blocked')),
                'post': safe(row.get('shots_post')),
                'goals': safe(row.get('shots_goals')),
                'xg_total': safe(row.get('shots_xg_total'), 2),
                'xgot_total': safe(row.get('shots_xgot_total'), 2),
                'left_foot': safe(row.get('shots_left_foot')),
                'right_foot': safe(row.get('shots_right_foot')),
                'header': safe(row.get('shots_header')),
                'inside_box': safe(row.get('shots_inside_box')),
            },
        }

        with open(OUT_DIR / f'{out_id}.json', 'w', encoding='utf-8') as f:
            json.dump(player, f, ensure_ascii=False, separators=(',', ':'))
        written += 1
        season_index.setdefault(pid, []).append({'id': out_id, 'season': season_label})

        # Every metric from every category (outfield + goalkeeping), as
        # [val, pct] pairs keyed by metric key — lets the leaderboard offer
        # a category tab (Attacking/Defending/.../Goalkeeping) and an "All"
        # view without a second per-player fetch. Null for a metric that
        # doesn't apply to this player's data (e.g. saves for an outfielder).
        cats = {}
        for cat_name, metric_list in LEADERBOARD_CATEGORIES.items():
            cat_out = {}
            for key, _label, _direction in metric_list:
                v = safe(row.get(f'{key}_val'), 2)
                p = safe(row.get(f'{key}_pct'))
                if v is not None or p is not None:
                    cat_out[key] = [v, p]
            if cat_out:
                cats[cat_name] = cat_out

        index.append({
            'id': out_id, 'base_id': pid, 'name': bio['name'], 'team': bio['team'], 'league': bio['league'],
            'season': bio['season'],
            'pos': bio['position'], 'pos_key': safe(pos_key), 'age': bio['age'], 'mv': bio['market_value_eur'],
            'rating': bio['fotmob_rating'], 'rating_pct': bio['fotmob_rating_pct'],
            'photo': bio['photo_url'],
            'mp': bio['matches_played'], 'mins': bio['minutes_played'],
            'cats': cats,
            'trend': trend_map.get(pid),
        })
    return index, written


def link_cross_season_profiles(season_index):
    """For a player who has both a current- and prior-season file, stamp each
    file's bio.other_seasons with the full list (id + season label) so the
    profile page can offer season-switcher chips without a second fetch."""
    linked_players = 0
    for pid, seasons in season_index.items():
        if len(seasons) < 2:
            continue
        seasons_sorted = sorted(seasons, key=lambda s: s['season'], reverse=True)
        linked_players += 1
        for s in seasons_sorted:
            fpath = OUT_DIR / f"{s['id']}.json"
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            data['bio']['other_seasons'] = seasons_sorted
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    return linked_players


def main():
    print('Building market value history files...')
    n_val, trend_map = build_valuation_files()
    print(f'  Wrote {n_val:,} valuation history files -> data/valuations/')

    print(f'Loading {SRC_CSV.name}...')
    df = pd.read_csv(SRC_CSV, low_memory=False)
    stale_mask = df['league'].isin(STALE_CURRENT_SEASON_LEAGUES) & (df['matches_played'] > STALE_CURRENT_SEASON_MP_THRESHOLD)
    if stale_mask.any():
        print(f'  Dropping {stale_mask.sum():,} current-season rows with implausible matches_played '
              f'(>{STALE_CURRENT_SEASON_MP_THRESHOLD}) — leftover prior-season data never refreshed after the season transition')
        df = df[~stale_mask]
    df = df.sort_values('matches_played', ascending=False).drop_duplicates('player_id', keep='first')
    print(f'  {len(df):,} unique players (current season)')

    season_index = {}
    index, written = process_dataframe(
        df, trend_map, season_index,
        out_id_fn=lambda pid: pid,
        season_label_fn=lambda row: safe(row.get('season')),
    )

    if PREV_SEASON_CSV.exists():
        print(f'Loading {PREV_SEASON_CSV.name}...')
        pdf = pd.read_csv(PREV_SEASON_CSV, low_memory=False)
        pdf = pdf[~pdf['league'].isin(PREV_SEASON_EXCLUDE_LEAGUES)]
        pdf = pdf[~pdf['player_id'].isin(PREV_SEASON_EXCLUDE_PLAYER_IDS)]
        elite_stale_mask = (
            (pdf['is_international'] == 1)
            & (pdf['matches_played'] <= PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MP)
            & (pdf['market_value_eur'] >= PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MV)
        )
        if elite_stale_mask.any():
            print(f'  Dropping {elite_stale_mask.sum():,} likely tournament-contaminated elite rows '
                  f'(international, <={PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MP} matches, '
                  f'>={PREV_SEASON_EXCLUDE_ELITE_LOW_MATCHES_MV/1e6:.0f}M value)')
            pdf = pdf[~elite_stale_mask]
        pdf = pdf.sort_values('matches_played', ascending=False).drop_duplicates('player_id', keep='first')
        print(f'  {len(pdf):,} unique players ({PREV_SEASON_LABEL}, after excluding calendar-year leagues)')
        prev_index, prev_written = process_dataframe(
            pdf, trend_map, season_index,
            out_id_fn=lambda pid: f'{pid}_{PREV_SEASON_LABEL}',
            season_label_fn=lambda row: PREV_SEASON_LABEL,
        )
        index.extend(prev_index)
        written += prev_written

    linked = link_cross_season_profiles(season_index)
    print(f'  Linked {linked:,} players who have data in more than one season')

    with open(WEBAPP / 'data' / 'fotmob_index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, separators=(',', ':'))

    # Single source of truth for the leaderboard's category tabs: which
    # metrics exist per category, their display label, and sort direction
    # (1 = higher is better, -1 = lower is better) — mirrors LEADERBOARD_CATEGORIES.
    cat_meta = {
        cat: [{'key': k, 'label': lbl, 'dir': d} for k, lbl, d in metrics]
        for cat, metrics in LEADERBOARD_CATEGORIES.items()
    }
    with open(WEBAPP / 'data' / 'fotmob_categories.json', 'w', encoding='utf-8') as f:
        json.dump(cat_meta, f, ensure_ascii=False, separators=(',', ':'))

    print(f'Wrote {written:,} player files -> {OUT_DIR}')
    print(f'Wrote index ({len(index):,} players) -> data/fotmob_index.json')


if __name__ == '__main__':
    main()
