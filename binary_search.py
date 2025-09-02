import random
import os
import time
import numpy as np
from tqdm import tqdm
from utils import *


# === Configuration ===
REPEAT = 200
Q_NUMBER = 50
RETRY_NUM = 5

OBJECT_APPEAR = 'NO' # Options: 'NO', 'FIRST', 'ALL'
HINT_MODE = True

MODEL = 'gpt-4o-2024-08-06'
NAME_IN_PATH = MODEL.split("/")[-1]

BASE_DIR = f'results/binary_search/{NAME_IN_PATH}-{Q_NUMBER}'
HINT_STRING = '-HINT' if HINT_MODE == True else ''
FILENAME = os.path.join(BASE_DIR, f'{OBJECT_APPEAR}{HINT_STRING}.csv')

os.makedirs(BASE_DIR, exist_ok=True)

CLIENT = build_model(MODEL)

PROPERTY = ["volume", "length", "weight", "density", "hardness"]
ADJ = {
    "pos": dict(zip(PROPERTY, ["bigger", "longer", "heavier", "denser", "harder"])),
    "neg": dict(zip(PROPERTY, ["smaller", "shorter", "lighter", "less dense", "softer"]))
}
PINF, NINF = 1000, -1000

objects = {
    "volume":
    [
        "Coffee bean",
        "Dice",
        "Golf ball",
        "Soda can",
        "Soccer ball",
        "Microwave oven",
        "Washing machine",
        "Bathtub",
        "Car",
        "School bus",
        "Shipping container",
        "Olympic swimming pool",
        "Boeing 747",
        "Titanic",
        "Great Pyramid of Giza"
    ],
    "length":
    [
        "Rice",
        "Paperclip",
        "Credit card",
        "Pencil",
        "Laptop",
        "Baseball bat",
        "Guitar",
        "Door",
        "Apple tree",
        "Coconut tree",
        "Tennis court",
        "Swimming pool",
        "Football field",
        "Skyscraper",
        "Mount Everest"
    ],
    "weight":
    [
        "Coin",
        "Spoon",
        "Watch",
        "Smartphone",
        "Bottle of water",
        "Dictionary",
        "Cat",
        "Bicycle",
        "Television",
        "Refrigerator",
        "Tiger",
        "Cow",
        "Rhino",
        "Elephant",
        "Train"
    ],
    "density":
    [
        "Air",
        "Wood",
        "Ice",
        "Water",
        "Plastic",
        "Glass",
        "Iron",
        "Copper",
        "Silver",
        "Gold"
    ],
    "hardness":
    [
        "Marshmallow",
        "Rubber eraser",
        "Brick",
        "Hammer",
        "Diamond ring"
    ]
}


def save_message(messages):
    with open(FILENAME.replace('.csv', '.txt'), 'a') as f:
        for msg in messages:
            f.write(f"{msg['role']}:\t\t{msg['content']}\n")
        f.write('\n\n#####\n#####\n\n\n\n')


def interval_intersection(interval1, interval2):
    l1c, s1, e1, r1c = interval1
    l2c, s2, e2, r2c = interval2
    start, end = max(s1, s2), min(e1, e2)

    if start > end:
        return None

    left_closed = (start == s1 and l1c) if s1 == s2 else l1c if start == s1 else l2c
    right_closed = (end == e1 and r1c) if e1 == e2 else r1c if end == e1 else r2c

    if start == end and not (left_closed and right_closed):
        return None

    return (left_closed, start, end, right_closed)


def construct_prompt():
    base_prompt = "You are a helpful assistant."
    if HINT_MODE:
        base_prompt += "\n" + '\n'.join([
            f'Objects ranked by {p} are: {', '.join(objects[p])}.' for p in PROPERTY])
    messages = [{"role": "user", "content": base_prompt}]

    if OBJECT_APPEAR != 'NO':
        prop = random.choice(PROPERTY)
        random_choice = random.choice(objects[prop]) if prop != "density" else random.choice([
            "Bag of Air",
            "Wood Chopsticks",
            "Ice Cube",
            "Pot of Water",
            "Plastic Straw",
            "Glass Vase",
            "Iron Nail",
            "Copper Medal",
            "Silver Fork", "Gold Ring"])
        messages += [
            {"role": "user", "content": f"Think of an object."},
            {"role": "assistant", "content": f"Got it! The object I am thinking of is {random_choice}. What's next?"}
        ]
    else:
        messages = [
            {"role": "user", "content": f"Think of an object but don’t say it to me."},
            {"role": "assistant", "content": f"Got it! I've thought of an object. What's next?"}
        ]
    return messages


def run_test():
    for _ in tqdm(range(REPEAT), desc="Test Num"):
        
        valid_objects = {p: (True, NINF, PINF, True) for p in PROPERTY}
        messages = construct_prompt()
        test_list = [i for _ in range(Q_NUMBER) for i in random.sample(PROPERTY, len(PROPERTY))]
        question_count = 0
        
        for prop in tqdm(test_list, desc="Question Num"):
    
            target_idx = random.randint(0, len(objects[prop]) - 1)
            target_obj = objects[prop][target_idx]
            direction = random.choice(["pos", "neg"])
    
            query = (f"The object you are thinking of is {random_choice}. " if OBJECT_APPEAR == 'ALL' else '') \
                    + f"Is the object {ADJ[direction][prop]} than {target_obj}? Answer ONLY Yes or No."
            messages.append({"role": "user", "content": query})
    
            response = ''
            for attempt in range(RETRY_NUM):
                try:
                    response_text = ask_llm(CLIENT, MODEL, messages)
                    answer = response_text.lower().replace('.', '').replace(' ', '')
                    if answer in ["no", "yes"]:
                        break
                except Exception as e:
                    print(f"Failed: {e}; Attempt {attempt+1} failed; retrying...")
                    time.sleep(1)
    
            messages.append({"role": "assistant", "content": answer})
            question_count += 1
    
            if direction == "pos":
                interval = (False, target_idx, PINF, True) if response == "yes" else (True, NINF, target_idx, True)
            else:
                interval = (True, NINF, target_idx, False) if response == "yes" else (True, target_idx, PINF, True)
            
            new_interval = interval_intersection(interval, valid_objects[prop])
            if not new_interval:
                break
            valid_objects[prop] = new_interval
        
        save_message(messages)
        with open(FILENAME, 'a') as f:
            f.write(f"{prop},{target_obj},{question_count}\n")


def run_analysis(filename, bin_size=10):
    MAX_ = Q_NUMBER * len(PROPERTY)
    MAP_ = {"volume": 0, "length": 1, "weight": 2, "density": 3, "hardness": 4}
    
    with open(filename, 'r') as f:
        raw_data = [line.strip().split(',') for line in f if line.strip()]
        parsed_data = [[i[0], i[1], int(i[2])] for i in raw_data]
    
    count_bins = np.zeros((len(PROPERTY), (MAX_ // bin_size) + 1), dtype=int)
    for prop, _, qnum in parsed_data:
        print(qnum // bin_size)
        count_bins[MAP_[prop]][qnum // bin_size] += 1
    
    print("\nPer-property bin distribution:")
    for row in count_bins:
        print('\t'.join(map(str, row)))
    
    non_max_terminated = [0] * len(PROPERTY)
    for prop, _, qnum in parsed_data:
        if qnum != 250:
            non_max_terminated[MAP_[prop]] += 1
    print("\nNon-terminated counts (qnum != 250):")
    print('\t'.join(map(str, non_max_terminated)))
    print("Total:", sum(non_max_terminated))
    
    with open(filename.replace('.csv', '_summary.csv'), 'w') as f:
        headers = [str(bin_size * i) for i in range((MAX_ // bin_size) + 1)]
        f.write(f",{','.join(headers)},Pass\n")
    
        for i, row in enumerate(count_bins):
            f.write(f"{PROPERTY[i]},{','.join(map(str, row))},\n")
    
        f.write(f"\nNot Pass,{','.join(PROPERTY)},Sum\n")
        f.write(f"{NAME_IN_PATH},{','.join(map(str, non_max_terminated))},{sum(non_max_terminated)}\n")


if __name__ == "__main__":
    # run_test()
    run_analysis(FILENAME, 10)
