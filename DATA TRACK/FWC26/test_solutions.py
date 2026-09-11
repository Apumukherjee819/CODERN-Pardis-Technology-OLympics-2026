import pandas as pd
import numpy as np
import sys
import io
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_PATH = 'C:/Users/arpam/OneDrive/Desktop/DATA OLYMPICS/FWC26/Data/'

df_results = pd.read_csv(DATA_PATH + 'results.csv')
df_goals = pd.read_csv(DATA_PATH + 'goalscorers.csv')
df_elo = pd.read_csv(DATA_PATH + 'elo_ratings_wc2026.csv')
df_fc = pd.read_csv(DATA_PATH + 'FC26.csv', low_memory=False)

df_results['date'] = pd.to_datetime(df_results['date'], format='mixed')
df_goals['date'] = pd.to_datetime(df_goals['date'], format='mixed')

print("Data loaded")

# ============================================================
# Challenge 1: Master Penalty Takers
# ============================================================
print("\n--- Challenge 1 ---")
penalty_goals = df_goals[df_goals['penalty'] == True]
penalty_specialists = penalty_goals.groupby('scorer').size().reset_index(name='total_penalty_goals')
penalty_specialists = penalty_specialists.sort_values(
    by=['total_penalty_goals', 'scorer'], ascending=[False, True]
).head(20).reset_index(drop=True)
penalty_specialists.columns = ['scorer_name', 'total_penalty_goals']
print(penalty_specialists.head(5))

# ============================================================
# Challenge 2: Undervalued Wonderkids
# ============================================================
print("\n--- Challenge 2 ---")
undervalued_wonderkids = df_fc[
    (df_fc['age'] <= 21) &
    (df_fc['value_eur'] > 10_000_000) &
    (df_fc['wage_eur'] < 20_000)
][['short_name', 'age', 'value_eur', 'wage_eur', 'club_name']].copy()
undervalued_wonderkids = undervalued_wonderkids.sort_values(
    by=['wage_eur', 'value_eur', 'short_name'], ascending=[True, False, True]
).reset_index(drop=True)
print(undervalued_wonderkids.head(5))

# ============================================================
# Challenge 3: Unconquerable Fortresses
# ============================================================
print("\n--- Challenge 3 ---")
home_wins = df_results[df_results['home_score'] > df_results['away_score']]
home_fortresses = home_wins.groupby('home_team').size().reset_index(name='total_home_wins')
home_fortresses = home_fortresses.sort_values(
    by=['total_home_wins', 'home_team'], ascending=[False, True]
).reset_index(drop=True)
print(home_fortresses.head(5))

# ============================================================
# Challenge 4: Continental Kings in 2025
# ============================================================
print("\n--- Challenge 4 ---")
elo_2025 = df_elo[df_elo['year'] == 2025]
idx = elo_2025.groupby('confederation')['rating'].idxmax()
confederation_kings = elo_2025.loc[idx, ['confederation', 'country', 'rating']].copy()
confederation_kings.columns = ['confederation', 'country', 'max_rating_in_2025']
confederation_kings = confederation_kings.sort_values(
    by=['max_rating_in_2025', 'country'], ascending=[False, True]
).reset_index(drop=True)
print(confederation_kings)

# ============================================================
# Challenge 5: Starting XI Strength
# ============================================================
print("\n--- Challenge 5 ---")
elo_2026 = df_elo[(df_elo['year'] == 2026) & (df_elo['snapshot_date'] == '12/31/2026')]
top20_elo = elo_2026[elo_2026['rank'] <= 20][['country', 'rank']].copy()
top20_elo.columns = ['country_name', 'elo_rank_2026']

results_xi = []
for _, row in top20_elo.iterrows():
    country = row['country_name']
    rank = row['elo_rank_2026']
    country_players = df_fc[df_fc['nationality_name'] == country]
    if len(country_players) >= 11:
        avg_overall = country_players.nlargest(11, 'overall')['overall'].mean()
    else:
        avg_overall = country_players['overall'].mean() if len(country_players) > 0 else 0
    results_xi.append({'country_name': country, 'elo_rank_2026': rank, 'avg_top11_overall': round(avg_overall, 2)})

starting_xi_strength = pd.DataFrame(results_xi).sort_values(
    by=['avg_top11_overall', 'country_name'], ascending=[False, True]
).reset_index(drop=True)
print(starting_xi_strength.head(5))

# ============================================================
# Challenge 6: Last-Minute Specialists
# ============================================================
print("\n--- Challenge 6 ---")
late_goals = df_goals[df_goals['minute'] >= 85]
late_count = late_goals.groupby('scorer').size().reset_index(name='late_goals_count')
late_count = late_count.sort_values(by=['late_goals_count', 'scorer'], ascending=[False, True]).head(20).reset_index(drop=True)

def make_fc26_key(name):
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return parts[0][0] + '. ' + parts[-1]
    return name

fc26_long = {}
fc26_short = {}
for _, row in df_fc.iterrows():
    ln, sn = row['long_name'], row['short_name']
    if pd.notna(ln) and ln not in fc26_long:
        fc26_long[ln] = (row['power_stamina'], row['overall'])
    if pd.notna(sn) and sn not in fc26_short:
        fc26_short[sn] = (row['power_stamina'], row['overall'])

def lookup_fc26(scorer):
    if scorer in fc26_long:
        return fc26_long[scorer]
    if scorer in fc26_short:
        return fc26_short[scorer]
    key = make_fc26_key(scorer)
    if key in fc26_short:
        return fc26_short[key]
    return (np.nan, np.nan)

late_count['power_stamina'] = late_count['scorer'].apply(lambda x: lookup_fc26(x)[0])
late_count['fc26_overall'] = late_count['scorer'].apply(lambda x: lookup_fc26(x)[1])
late_goal_specialists = late_count
print(late_goal_specialists.head(5))

# ============================================================
# Challenge 7: High-Scoring Cities of the 2020s
# ============================================================
print("\n--- Challenge 7 ---")
results_2020s = df_results[df_results['date'].dt.year >= 2020].copy()
results_2020s['total_goals'] = results_2020s['home_score'] + results_2020s['away_score']
city_stats = results_2020s.groupby('city').agg(
    total_matches_hosted=('total_goals', 'count'),
    total_goals_scored=('total_goals', 'sum')
).reset_index()
city_stats = city_stats[city_stats['total_matches_hosted'] >= 10]
city_stats['avg_goals_per_match'] = (city_stats['total_goals_scored'] / city_stats['total_matches_hosted']).round(2)
high_scoring_cities = city_stats.sort_values(
    by=['avg_goals_per_match', 'city'], ascending=[False, True]
).reset_index(drop=True)
print(high_scoring_cities.head(5))

# ============================================================
# Challenge 8: Club Footprints in International Matches
# ============================================================
print("\n--- Challenge 8 ---")
scorer_to_club = {}
for _, row in df_fc.iterrows():
    ln, sn, club = row['long_name'], row['short_name'], row['club_name']
    if pd.notna(ln) and ln not in scorer_to_club:
        scorer_to_club[ln] = club
    if pd.notna(sn) and sn not in scorer_to_club:
        scorer_to_club[sn] = club

df_goals_valid = df_goals[df_goals['scorer'].notna()].copy()
df_goals_valid['club_name'] = df_goals_valid['scorer'].map(scorer_to_club)
goals_with_club = df_goals_valid[df_goals_valid['club_name'].notna()]

club_contributions = goals_with_club.groupby('club_name').agg(
    unique_international_scorers=('scorer', 'nunique'),
    total_international_goals_by_squad=('scorer', 'count')
).reset_index()
club_contributions = club_contributions.sort_values(
    by=['total_international_goals_by_squad', 'club_name'], ascending=[False, True]
).reset_index(drop=True)
print(club_contributions.head(5))

# ============================================================
# Challenge 9: Game-Changing Goals
# ============================================================
print("\n--- Challenge 9 ---")

# Build elo year-end lookup
elo_yearly = df_elo[df_elo['snapshot_date'].str.contains('12/31')].copy()
elo_yearly['year'] = elo_yearly['year'].astype(int)
elo_lookup = {}
for _, row in elo_yearly.iterrows():
    elo_lookup[(row['country'], row['year'])] = row['rating']

# Merge goals with results to get home/away info
goals_merged = df_goals.merge(
    df_results[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
    on=['date', 'home_team', 'away_team'], how='inner'
).sort_values(['date', 'home_team', 'away_team', 'minute'])

# For each match group, find game-changing goals efficiently
def process_match_group(grp):
    results = []
    home_team = grp['home_team'].iloc[0]
    away_team = grp['away_team'].iloc[0]
    final_hs = grp['home_score'].iloc[0]
    final_as = grp['away_score'].iloc[0]
    
    hg, ag = 0, 0
    goals_list = []
    for _, g in grp.iterrows():
        team = g['team']
        if team == home_team:
            hg += 1
        else:
            ag += 1
        goals_list.append((g['scorer'], team, g['minute'], hg, ag))
    
    # Find equalizers
    for scorer, team, minute, h, a in goals_list:
        if team == home_team:
            stg, og = h, a
        else:
            stg, og = a, h
        stg_before = stg - 1
        og_before = og if team != home_team else og
        # Actually let me recalculate properly
        pass
    
    # Redo with before/after tracking
    hg, ag = 0, 0
    for scorer, team, minute, h, a in goals_list:
        hg_before, ag_before = hg, ag
        if team == home_team:
            hg += 1
        else:
            ag += 1
        
        if team == home_team:
            my_before, their_before = hg_before, ag_before
            my_after, their_after = hg, ag
        else:
            my_before, their_before = ag_before, hg_before
            my_after, their_after = ag, hg
        
        if my_before < their_before and my_after == their_after:
            results.append({'scorer': scorer, 'type': 'equalizer'})
    
    # Find winning goal
    if final_hs > final_as:
        winner = home_team
    elif final_as > final_hs:
        winner = away_team
    else:
        return results
    
    hg, ag = 0, 0
    for i, (scorer, team, minute, h, a) in enumerate(goals_list):
        if team == home_team:
            hg += 1
        else:
            ag += 1
        
        if winner == home_team:
            ahead = hg > ag
        else:
            ahead = ag > hg
        
        if ahead:
            temp_hg, temp_ag = hg, ag
            stays = True
            for j in range(i+1, len(goals_list)):
                _, ft, _, _, _ = goals_list[j]
                if ft == home_team:
                    temp_hg += 1
                else:
                    temp_ag += 1
                if winner == home_team and temp_hg <= temp_ag:
                    stays = False
                    break
                elif winner == away_team and temp_ag <= temp_hg:
                    stays = False
                    break
            if stays:
                results.append({'scorer': scorer, 'type': 'winner'})
                break
    
    return results

match_groups = goals_merged.groupby(['date', 'home_team', 'away_team'])
all_gc = []
for keys, grp in match_groups:
    gcs = process_match_group(grp)
    for gc in gcs:
        all_gc.append({'scorer': gc['scorer'], 'type': gc['type']})

gc_df = pd.DataFrame(all_gc)
print(f"Game-changing goals: {len(gc_df)}")

player_gc = gc_df.groupby('scorer').agg(
    total_game_changing_goals=('type', 'count')
).reset_index()

player_years = df_goals.merge(
    df_results[['date', 'home_team', 'away_team']], 
    on=['date', 'home_team', 'away_team']
).merge(gc_df[['scorer']], on='scorer', how='inner').groupby('scorer')['date'].apply(
    lambda x: list(x.dt.year.unique())
).reset_index()
player_years.columns = ['scorer', 'years']

fc26_nat = {}
fc26_ovr = {}
for _, row in df_fc.iterrows():
    ln, sn = row['long_name'], row['short_name']
    if pd.notna(ln) and ln not in fc26_nat:
        fc26_nat[ln] = row['nationality_name']
        fc26_ovr[ln] = row['overall']
    if pd.notna(sn) and sn not in fc26_nat:
        fc26_nat[sn] = row['nationality_name']
        fc26_ovr[sn] = row['overall']

player_gc['nationality_name'] = player_gc['scorer'].map(fc26_nat)
player_gc['fc26_overall_rating'] = player_gc['scorer'].map(fc26_ovr)
player_gc = player_gc.merge(player_years, on='scorer', how='left')

def get_elo_improvement(country, years):
    if pd.isna(country):
        return 'No Elo Data'
    improvements = []
    for year in sorted(years):
        cur = elo_lookup.get((country, year))
        prev = elo_lookup.get((country, year - 1))
        if cur is not None and prev is not None:
            improvements.append(cur - prev)
    if not improvements:
        return 'No Elo Data'
    return round(np.mean(improvements), 2)

player_gc['avg_elo_improvement'] = player_gc.apply(
    lambda r: get_elo_improvement(r['nationality_name'], r['years']) if isinstance(r['years'], list) else 'No Elo Data',
    axis=1
)

game_changing_goals = player_gc[['scorer', 'nationality_name', 'total_game_changing_goals', 'avg_elo_improvement', 'fc26_overall_rating']].copy()
game_changing_goals.columns = ['scorer_short_name', 'nationality_name', 'total_game_changing_goals', 'avg_elo_improvement', 'fc26_overall_rating']
game_changing_goals = game_changing_goals.sort_values(by='scorer_short_name', ascending=True).reset_index(drop=True)
print(game_changing_goals.head(5))

# ============================================================
# Challenge 10: Paradox of Potential and Team Decline
# ============================================================
print("\n--- Challenge 10 ---")

elo_yearly_sorted = elo_yearly.sort_values(['country', 'year'])
elo_yearly_sorted['elo_change'] = elo_yearly_sorted.groupby('country')['rating'].diff()
big_drops = elo_yearly_sorted[elo_yearly_sorted['elo_change'] < -50].copy()
big_drops['elo_drop_amount'] = big_drops['elo_change'].abs()

avg_potential = df_fc.groupby('nationality_name')['potential'].mean().reset_index()
avg_potential.columns = ['country_name', 'avg_fc26_potential']
potential_threshold = avg_potential['avg_fc26_potential'].quantile(0.90)
top10_set = set(avg_potential[avg_potential['avg_fc26_potential'] >= potential_threshold]['country_name'])

df_results['year'] = df_results['date'].dt.year
losses_list = []
for _, row in df_results.iterrows():
    hs, as_ = row['home_score'], row['away_score']
    if hs > as_:
        losses_list.append({'year': row['year'], 'country': row['away_team']})
    elif as_ > hs:
        losses_list.append({'year': row['year'], 'country': row['home_team']})

losses_per_year = pd.DataFrame(losses_list).groupby(['year', 'country']).size().reset_index(name='total_losses_in_year')

big_drops_merged = big_drops.merge(avg_potential, left_on='country', right_on='country_name', how='left')
big_drops_merged = big_drops_merged[big_drops_merged['country'].isin(top10_set)]
big_drops_merged = big_drops_merged.merge(losses_per_year, on=['year', 'country'], how='left')

underperforming_golden_generations = big_drops_merged[['country', 'year', 'elo_drop_amount', 'avg_fc26_potential', 'total_losses_in_year']].copy()
underperforming_golden_generations.columns = ['country_name', 'year_of_drop', 'elo_drop_amount', 'avg_fc26_potential', 'total_losses_in_year']
underperforming_golden_generations = underperforming_golden_generations.sort_values(
    by=['elo_drop_amount', 'country_name'], ascending=[False, True]
).reset_index(drop=True)
print(underperforming_golden_generations)

# ============================================================
# Challenge 11: Giant Killers
# ============================================================
print("\n--- Challenge 11 ---")

wc_results = df_results[df_results['tournament'] == 'FIFA World Cup'].copy()
wc_match_keys = set(zip(wc_results['date'], wc_results['home_team'], wc_results['away_team']))

wc_goals = df_goals[
    (df_goals['penalty'] == False) &
    (df_goals.apply(lambda r: (r['date'], r['home_team'], r['away_team']) in wc_match_keys, axis=1))
].copy()

# Get match years from results
wc_results_copy = wc_results.copy()
wc_results_copy['year'] = wc_results_copy['date'].dt.year
wc_year_lookup = wc_results_copy.set_index(['date', 'home_team', 'away_team'])['year'].to_dict()

fc26_val = {}
for _, row in df_fc.iterrows():
    ln, sn = row['long_name'], row['short_name']
    if pd.notna(ln) and ln not in fc26_val:
        fc26_val[ln] = row['value_eur']
    if pd.notna(sn) and sn not in fc26_val:
        fc26_val[sn] = row['value_eur']

giant_killer_rows = []
for _, goal in wc_goals.iterrows():
    match_key = (goal['date'], goal['home_team'], goal['away_team'])
    match_year = wc_year_lookup.get(match_key)
    if match_year is None:
        continue
    
    home_team, away_team = goal['home_team'], goal['away_team']
    scorer_team = goal['team']
    opponent_team = away_team if scorer_team == home_team else home_team
    
    elo_scorer = elo_lookup.get((scorer_team, match_year), elo_lookup.get((scorer_team, match_year - 1)))
    elo_opponent = elo_lookup.get((opponent_team, match_year), elo_lookup.get((opponent_team, match_year - 1)))
    
    if elo_scorer is not None and elo_opponent is not None:
        elo_diff = elo_opponent - elo_scorer
        if elo_diff >= 300:
            val = fc26_val.get(goal['scorer'], 'Not in FC26')
            giant_killer_rows.append({
                'giant_killer_name': goal['scorer'],
                'player_team': scorer_team,
                'opponent_team': opponent_team,
                'elo_difference': elo_diff,
                'goal_minute': goal['minute'],
                'player_value_eur': val
            })

giant_killers = pd.DataFrame(giant_killer_rows)
if len(giant_killers) > 0:
    giant_killers = giant_killers.sort_values(by='giant_killer_name', ascending=True).reset_index(drop=True)
print(giant_killers)
print(f"Total: {len(giant_killers)}")

# ============================================================
# Challenge 12: Elite Winning Streaks
# ============================================================
print("\n--- Challenge 12 ---")

non_friendly = df_results[df_results['tournament'] != 'Friendly'].copy()
non_friendly['date'] = pd.to_datetime(non_friendly['date'], format='mixed')
non_friendly['year'] = non_friendly['date'].dt.year

top10_per_year = {}
for year in elo_yearly['year'].unique():
    year_data = elo_yearly[elo_yearly['year'] == year].nsmallest(10, 'rank')
    top10_per_year[year] = set(year_data['country'].values)

all_streaks = []
all_teams = set(non_friendly['home_team'].unique()) | set(non_friendly['away_team'].unique())

for team in all_teams:
    home_m = non_friendly[non_friendly['home_team'] == team][['date', 'year', 'away_team', 'home_score', 'away_score']].copy()
    home_m['opponent'] = home_m['away_team']
    home_m['team_goals'] = home_m['home_score']
    home_m['opp_goals'] = home_m['away_score']
    
    away_m = non_friendly[non_friendly['away_team'] == team][['date', 'year', 'home_team', 'home_score', 'away_score']].copy()
    away_m['opponent'] = away_m['home_team']
    away_m['team_goals'] = away_m['away_score']
    away_m['opp_goals'] = away_m['home_score']
    
    team_matches = pd.concat([home_m, away_m]).sort_values('date').reset_index(drop=True)
    if len(team_matches) == 0:
        continue
    
    best_len = 0
    best_start = None
    best_end = None
    best_goals = 0
    cur_len = 0
    cur_start = None
    cur_goals = 0
    
    for _, m in team_matches.iterrows():
        is_win = m['team_goals'] > m['opp_goals']
        opp_in_top10 = m['opponent'] in top10_per_year.get(m['year'], set())
        
        if is_win and opp_in_top10:
            if cur_len == 0:
                cur_start = m['date']
            cur_len += 1
            cur_goals += m['team_goals']
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = team_matches.iloc[_ - 1]['date'] if cur_len > 0 else None
                best_goals = cur_goals
            cur_len = 0
            cur_start = None
            cur_goals = 0
    
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
        best_end = team_matches.iloc[-1]['date'] if cur_len > 0 else None
        best_goals = cur_goals
    
    if best_len > 0:
        all_streaks.append({
            'team_name': team,
            'max_win_streak_vs_top10': best_len,
            'streak_start_date': best_start,
            'streak_end_date': best_end,
            'total_goals_scored_in_streak': best_goals
        })

elite_winning_streaks = pd.DataFrame(all_streaks)
if len(elite_winning_streaks) > 0:
    elite_winning_streaks = elite_winning_streaks.sort_values(
        by=['max_win_streak_vs_top10', 'team_name'], ascending=[False, True]
    ).reset_index(drop=True)
print(elite_winning_streaks.head(10))

print("\n=== ALL CHALLENGES COMPLETE ===")
