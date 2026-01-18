import pandas as pd
from datetime import timedelta
import random
import os
import numpy as np
import multiprocessing

import time
from tqdm import tqdm

N_TOP_FREQUENT = 3 
TIME_LIMITS = {
    'month': timedelta(days=30),
    'week': timedelta(days=7),
    'day': timedelta(days=1)
}

def get_fuzzy_time_expression(time_delta, target_time):
    if time_delta > TIME_LIMITS['month']:
        months = round(time_delta.days / 30)
        return f"in about {months} months" if months > 1 else "in the next month"
    
    elif time_delta > TIME_LIMITS['week']:
        weeks = round(time_delta.days / 7)
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_time.weekday()]
        return f"in about {weeks} weeks, specifically next {weekday}"
    
    elif time_delta > TIME_LIMITS['day']:
        days = time_delta.days
        return f"in about {days} days, on {target_time.date().strftime('%Y-%m-%d')}"
    
    else: 
        hour = target_time.hour
        if 5 <= hour < 12:
            period = "in the morning"
        elif 12 <= hour < 18:
            period = "in the afternoon"
        elif 18 <= hour < 22:
            period = "in the evening"
        else:
            period = "late at night"
        return f"later today, {period}"


def generate_context_for_df(df_input: pd.DataFrame, N_top_frequent: int,
                            StaySavePath: str) -> None:

    df = df_input.copy()
    
    df['stime'] = pd.to_datetime(df['stime'], format="%Y-%m-%d %H:%M:%S")
    df['etime'] = pd.to_datetime(df['etime'], format="%Y-%m-%d %H:%M:%S")
    df['context_fuzzy'] = None  
    df['context_precise'] = None  
    
    if len(df) < N_top_frequent or df.empty:
        return df

    top_n_grids = df['grid'].value_counts().nlargest(N_top_frequent).index.tolist()
    # print(top_n_grids)
    aperiodic_stay_list = df[~df['grid'].isin(top_n_grids)].index.tolist()
    current_stay_candidates = df.index[:-1].tolist()

    while aperiodic_stay_list and current_stay_candidates:
        current_stay_idx = random.choice(current_stay_candidates)
        
        future_aperiodic_stays = [
            target_idx for target_idx in aperiodic_stay_list 
            if target_idx > current_stay_idx
        ]
        
        if not future_aperiodic_stays:
            current_stay_candidates.remove(current_stay_idx)
            continue

        generate_context_stay_idx = random.choice(future_aperiodic_stays)
        
        current_stay = df.loc[current_stay_idx]
        generate_context_stay = df.loc[generate_context_stay_idx]
        
        start_grid = current_stay['grid']
        end_grid = generate_context_stay['grid']
        user_id = current_stay['userID']
        
        time_delta = generate_context_stay['stime'] - current_stay['etime']
        
        precise_time_str = generate_context_stay['stime'].strftime('%Y-%m-%d %H:%M:%S')
        context_precise = (
            f"User {user_id} will move from grid {start_grid} to grid {end_grid}, "
            f"at {precise_time_str}."
        )
        
        fuzzy_time_expression = None
        should_use_fuzzy = False

        if time_delta > TIME_LIMITS['month']:
            if random.random() >= 0.1: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        elif time_delta > TIME_LIMITS['week']:
            if random.random() >= 0.3: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        elif time_delta > TIME_LIMITS['day']:
            if random.random() >= 0.5: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        else: 
            if random.random() >= 0.7: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True

        if should_use_fuzzy and fuzzy_time_expression:
            context_fuzzy = (
                f"User {user_id} will move from grid {start_grid} to grid {end_grid}, "
                f"arriving around {fuzzy_time_expression}."
            )
        else:
            context_fuzzy = context_precise

        df.loc[current_stay_idx, 'context_fuzzy'] = context_fuzzy
        df.loc[current_stay_idx, 'context_precise'] = context_precise
        
        aperiodic_stay_list.remove(generate_context_stay_idx)
        current_stay_candidates.remove(current_stay_idx)

        df.to_csv(StaySavePath+f"{user_id}.csv", index=True)
        print(f'--- {user_id}')
    


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def merge_csvs(folder_path:str, output_path:str) -> None:

    all_files = get_all_file_paths(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    if not csv_files:
        print("No CSV file found.")
        return
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully merged {len(csv_files)} CSV files and saved them to {output_path}")


if __name__ == "__main__":


    start_time = time.time()

    ProcessManager = multiprocessing.Manager()

    ProcessPool = multiprocessing.Pool()

    gUserTrajPath = './Data/MoreUser/Input/'
    OutputStayPath = "./Data/MoreUser/Output/"
    OutputAllStayPath = "./Data/MoreUser/all.csv"
    


    all_files = get_all_file_paths(gUserTrajPath)

    N_top_frequent = 3

    for singleuserfile in tqdm(all_files):
        singleUserDf = pd.read_csv(singleuserfile, index_col=0)

        ProcessPool.apply_async(generate_context_for_df, 
                                    args=(singleUserDf, N_top_frequent,
                                            OutputStayPath))
    ProcessPool.close()
    ProcessPool.join()
    ProcessManager.shutdown()

    print("All user data has been processed; data merging will begin....")
    merge_csvs(OutputStayPath, OutputAllStayPath)

    print(f"All data processing completed, total time elapsed {time.time() - start_time:.2f} 秒")