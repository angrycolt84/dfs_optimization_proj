# ASSUME DATA LOADS IN AS A DATAFRAME

# TODO: OPTIMIZE DATAFRAME TOTALS TO CREATE LINEUPS USING CONFIGURABLE BUDGET

# TODO: ADD RANDOMNESS FACTOR FOR DFS LINEUPS

# TODO: ADD ABILITY TO GENERATE N NUMBER OF RANDOM LINEUPS

# TODO: CREATE EXPORT THAT CAN CALL API TO UPLOAD LINEUPS?


import pandas as pd
import numpy as np
from pulp import *

def stochastic_optimizer(df, salary_cap, num_simulations=100, variance=0.15):
    """
    Generate lineups using stochastic sampling of projections
    
    Parameters:
    - df: DataFrame with columns [player, position, salary, projected_points]
    - salary_cap: Max salary for lineup
    - num_simulations: Number of Monte Carlo simulations
    - variance: % variance in projections (e.g., 0.15 = ±15%)
    """
    lineups = []
    
    for sim in range(num_simulations):
        # Sample projections from normal distribution
        df['sampled_points'] = np.random.normal(
            df['projected_points'],
            df['projected_points'] * variance
        )
        
        # Optimize using sampled projections
        lineup = optimize_lineup(df, salary_cap, use_col='sampled_points')
        lineups.append(lineup)
    
    return lineups

def optimize_lineup(df, salary_cap, use_col='projected_points'):
    """Standard linear optimization for a single lineup"""
    prob = LpProblem("DFS_Optimizer", LpMaximize)
    
    # Decision variables
    players = {i: LpVariable(f"player_{i}", cat='Binary') for i in df.index}
    
    # Objective: maximize points
    prob += lpSum([players[i] * df.loc[i, use_col] for i in df.index])
    
    # Constraints
    prob += lpSum([players[i] * df.loc[i, 'salary'] for i in df.index]) <= salary_cap
    
    # Position constraints (adjust as needed)
    for pos in df['position'].unique():
        pos_mask = df['position'] == pos
        prob += lpSum([players[i] for i in df[pos_mask].index]) >= 1
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    return df[df.index.isin([i for i in players if players[i].varValue == 1])]



def game_theory_optimizer(df, salary_cap, num_lineups=20, diversity_weight=0.3):
    """
    Generate diverse lineups to maximize tournament edge
    
    Parameters:
    - diversity_weight: Balance between optimal vs diverse (0-1)
    """
    lineups = []
    player_usage = pd.Series(0, index=df.index)  # Track player picks
    
    for lineup_num in range(num_lineups):
        # Adjust points based on exposure (penalize overused players)
        df['adjusted_points'] = df['projected_points'] * (1 - diversity_weight * player_usage / lineup_num + 1)
        
        lineup = optimize_lineup(df, salary_cap, use_col='adjusted_points')
        lineups.append(lineup)
        
        # Update usage for next iteration
        player_usage[lineup.index] += 1
    
    return lineups

def analyze_lineup_diversity(lineups):
    """Measure overlap between generated lineups"""
    overlaps = []
    for i in range(len(lineups)):
        for j in range(i+1, len(lineups)):
            overlap = len(set(lineups[i].index) & set(lineups[j].index))
            overlaps.append(overlap)
    
    return {
        'avg_overlap': np.mean(overlaps),
        'max_overlap': np.max(overlaps),
        'min_overlap': np.min(overlaps)
    }