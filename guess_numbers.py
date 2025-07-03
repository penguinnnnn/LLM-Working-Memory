import numpy as np
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from utils import *


# === Configuration ===
RANGE = 10
REPEAT = 200
RETRY_NUM = 5
MODEL = 'gpt-4o-2024-08-06'
NAME_IN_PATH = MODEL.split("/")[-1]
BASE_DIR = f'results/guess_numbers/{NAME_IN_PATH}-{RANGE}'

os.makedirs(BASE_DIR, exist_ok=True)

CLIENT = build_model(MODEL)

count_yes = np.zeros(RANGE, dtype=int)
count_all = np.zeros(RANGE, dtype=int)

prob_matrix = []

for _ in tqdm(range(REPEAT), desc="Repeats"):
    shuffled_array = np.random.permutation(np.arange(RANGE))
    response_records = []
    
    for index, i in enumerate(shuffled_array):
        answer = ''
        msgs = [
            {"role": "user", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Think of an integer between 1 and {RANGE}, but don’t say it to me."},
            {"role": "assistant", "content": f"Got it! I've thought of an integer between 1 and {RANGE}. What's next?"},
            {"role": "user", "content": f"Is the number you’re thinking of {i + 1}? Answer Yes or No in the following JSON format: {{\"answer\": \"yes_or_no\"}}."}
        ]
        for attempt in range(RETRY_NUM):
            try:
                response_text = ask_llm(CLIENT, MODEL, msgs)
                response = json.loads(extract_last_json(response_text))
                answer = response.get("answer", "").lower()
                if answer in ["no", "yes"]:
                    break
            except Exception as e:
                print(f"Failed to parse JSON: {e}; Attempt {attempt+1} failed; retrying...")
                time.sleep(1)

        response_records.append((i + 1, answer))
        
        if answer == "yes":
            count_yes[i] += 1

        count_all[i] += 1

    with open(os.path.join(BASE_DIR, 'responses.csv'), 'a') as f:
        for digit, ans in response_records:
            f.write(f"{digit},{ans}\n")

    prob_dist = [count_yes[k] / count_all[k] if count_all[k] > 0 else 0.0 for k in range(RANGE)]
    prob_matrix.append(prob_dist)

prob_df = pd.DataFrame(prob_matrix, columns=[f"{i+1}" for i in range(RANGE)])
prob_df.to_csv(os.path.join(BASE_DIR, 'distribution.csv'), index=False)

print("\nFinal distribution:")
print(prob_df.tail(1))
