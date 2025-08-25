import json
import random
import os
import time
from tqdm import tqdm
from utils import *


# === Configuration ===
NUMBER = 10
REPEAT = 150
RETRY_NUM = 10
COT = False
MODEL = 'gpt-4o-2024-08-06'

NAME_IN_PATH = MODEL[MODEL.find('/') + 1:]
BASE_DIR = f'results/math_magic/{NAME_IN_PATH}-{NUMBER}'
cot_string = 'CoT-' if COT else ''
PATH = os.path.join(BASE_DIR, f'{cot_string}.csv')

os.makedirs(BASE_DIR, exist_ok=True)

CLIENT = build_model(MODEL)


def construct_prompt():
    messages = [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Think of 4 integers between 1 and {NUMBER} in order, but don't tell me."},
        {"role": "assistant", "content": f"Okay! I've got 4 numbers. What’s next?"},
        {"role": "user", "content": f"In order, append the same 4 numbers after the original ones."},
        {"role": "assistant", "content": f"Understood! Now I have 8 numbers. What’s next?"},
    ]
    
    random_number = random.randint(1, 7)

    messages.append({"role": "user", "content": f"Take the first {random_number} numbers and move them to the end."})
    messages.append({"role": "assistant", "content": f"Got it! Now I have moved the numbers. What’s next?"})
    
    messages.append({"role": "user", "content": f"Take the first 3 numbers and insert them anywhere in the middle."})
    messages.append({"role": "assistant", "content": f"Okay! The first 3 numbers are placed somewhere in the middle. What’s next?"})
    
    messages.append({"role": "user", "content": f"Set the first number aside. We don't need it for now."})
    messages.append({"role": "assistant", "content": f"Understood! Now I have 7 numbers. What’s next?"})
    
    random_number = random.randint(1, 3)
    
    messages.append({"role": "user", "content": f"Take the first {random_number} numbers and insert them anywhere in the middle."})
    messages.append({"role": "assistant", "content": f"Got it. The first {random_number} numbers are placed somewhere in the middle. What’s next?"})
    
    random_number = random.randint(1, 3)
    
    messages.append({"role": "user", "content": f"Remove the first {random_number} numbers. We will never need it anymore."})
    messages.append({"role": "assistant", "content": f"Okay! Now I have {7 - random_number} numbers. What’s next?"})
    
    messages.append({"role": "user", "content": f"Take the first number and move it to the end. Repeat this seven times."})
    messages.append({"role": "assistant", "content": f"Understood! Now my sequence has rearranged. What’s next?"})
    
    messages.append({"role": "user", "content": f"Take the first number and move it to the end, then remove the second number. Repeat this {6 - random_number} times."})
    messages.append({"role": "assistant", "content": f"Got it! Now I have only 1 number. What’s next?"})

    last_prompt = "Tell me what the last remaining number is. Do you remember the number you set aside at the beginning? Tell me what that number was. You can only reply in JSON format: {\"final_number\": \"a_number_from_1_to_" + str({NUMBER}) + "\", \"put_aside_number\": \"a_number_from_1_to_" + str({NUMBER}) + "\"}"
    if COT:
        last_prompt.replace("You can only reply", "Let's think step by step. Output your analysis first. At the end, reply")
    messages.append({"role": "user", "content": last_prompt})

    return messages


def save_message(messages):
    with open(PATH.replace(".csv", "response.txt"), 'a') as f:
        for msg in messages:
            f.write(f"{msg['role']}:\t\t{msg['content']}\n")
        f.write('\n\n#####\n#####\n\n\n\n')


for _ in tqdm(range(REPEAT), desc="Repeats"):
    
    final_number = NUMBER + 1
    put_aside_number = NUMBER + 2
    
    messages = construct_prompt()
    for retry_num in range(RETRY_NUM):
        try:
            response_text = ask_llm(CLIENT, MODEL, messages)
            response = json.loads(extract_last_json(response_text))
            final_number = int(response.get("final_number", NUMBER + 1))
            put_aside_number = int(response.get("put_aside_number", NUMBER + 2))
            success = final_number <= NUMBER and put_aside_number <= NUMBER
            if success:
                break
        except Exception as e:
            print(f"Failed to parse JSON: {e}; Attempt {retry_num+1} failed; retrying...")
            time.sleep(1)
    
    messages.append({"role": "assistant", "content": response_text})
    save_message(messages)
    with open(PATH.replace('.csv', 'result.csv'), 'a') as f:
        is_equal = final_number == put_aside_number
        contains_7 = is_equal and ("7" in str(final_number))
        f.write(f"{final_number},{put_aside_number},{is_equal},{contains_7}\n")
