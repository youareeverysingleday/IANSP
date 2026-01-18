import re
from datetime import datetime, timedelta
import math

import torch
import torch.nn as nn

DATA_PATH = "./Data/all_users_context_combined.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_context(context_str: str, reference_time: datetime)-> dict:

    if reference_time is None:
        reference_time = datetime.now()

    if not isinstance(context_str, str) or "will move" not in context_str:
        return None
    
    pattern = r"User\s+(\d+)\s+will\s+move\s+from\s+grid\s+(\d+)\s+to\s+grid\s+(\d+),\s*arriving\s+around\s+(.+)\."

    member = re.match(pattern, context_str.strip())
    if not member:
        return None

    user_id = int(member.group(1))
    start_grid = int(member.group(2))
    end_grid = int(member.group(3))
    time_expr = member.group(4).strip()

    time_text = None
    sigma_minute = None

    for fmt, default_sigma in [("%Y-%m-%d %H:%M:%S", 15.0), ("%Y-%m-%d", 60.0)]:
        try:
            # print(fmt, default_sigma)
            dt = datetime.strptime(time_expr, fmt)
            time_text = dt
            sigma_minute = default_sigma
            break
        except Exception:
            pass

    delta_minute = 0 
    if time_text is None:
        relative_pattern = r"(?:in\s+)?(?:about\s+)?([+-]?\d+)\s*(months?|weeks?|days?|hours?)"
        member_relative_time = re.search(relative_pattern, time_expr, flags=re.IGNORECASE)

        if member_relative_time:
            num = int(member_relative_time.group(1))
            unit = member_relative_time.group(2).lower()

            if unit.startswith("month"):
                days = num * 30
                delta_minute = days * 24 * 60
            elif unit.startswith("week"):
                days = num * 7
                delta_minute = days * 24 * 60
            elif unit.startswith("day"):
                days = num
                delta_minute = days * 24 * 60
            elif unit.startswith("hour"):
                delta_minute = num * 60
            else:
                delta_minute = num * 24 * 60

            time_text = reference_time + timedelta(minutes=delta_minute)

            if delta_minute >= 30 * 24 * 60:
                sigma_minute = max(24 * 60.0, 0.5 * delta_minute)

            elif delta_minute >= 7 * 24 * 60:
                sigma_minute = max(60.0, 0.5 * delta_minute)

            elif delta_minute >= 24 * 60:
                sigma_minute = max(30.0, 0.5 * delta_minute) 
            else:
                sigma_minute = max(15.0, 0.5 * delta_minute)  
        else:
            member_relative_time_candidate = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", time_expr)
            if member_relative_time_candidate:
                try:
                    dt = datetime.strptime(member_relative_time_candidate.group(1), "%Y-%m-%d %H:%M:%S")
                    time_text = dt
                    sigma_minute = 15.0
                except:
                    time_text = reference_time
                    sigma_minute = 24 * 60 * 30.0 
            else:
                time_text = reference_time
                sigma_minute = 24 * 60 * 30.0 
    
    if time_text is None:
        time_text = reference_time
    if sigma_minute is None:
        sigma_minute = 24 * 60 * 30.0

    return {
        "user_id": user_id,
        "start_grid": start_grid,
        "end_grid": end_grid,
        "time_text_minute": time_text.timestamp() / 60.0, 
        "sigma_minute": float(sigma_minute)
    }