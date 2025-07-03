import os
import time
from utils import *


# === Configuration ===
REPEAT = 100
RETRY_NUM = 5
MODEL = 'gpt-4o-mini-2024-07-18'
NAME_IN_PATH = MODEL[MODEL.find('/') + 1:]
PATH = f'{NAME_IN_PATH}-BASIC-ACCURACY.csv'

BASE_DIR = f'results/binary_search/'
FILENAME = os.path.join(BASE_DIR, f'{NAME_IN_PATH}-BASIC-ACCURACY.csv')

os.makedirs(BASE_DIR, exist_ok=True)

CLIENT = build_model(MODEL)

PROPERTY = ["volume", "length", "weight", "density", "hardness"]
ADJ = {
    "pos": dict(zip(PROPERTY, ["bigger", "longer", "heavier", "denser", "harder"])),
    "neg": dict(zip(PROPERTY, ["smaller", "shorter", "lighter", "less dense", "softer"]))
}

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

count = 0
correct = 0
for prop in PROPERTY:
    for obj1 in range(len(objects[prop])):
        for obj2 in range(len(objects[prop])):
            if obj1 == obj2: continue
            for adj in ['pos', 'neg']:

                messages = [
                    {"role": "user", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Is {objects[prop][obj1]} {ADJ[adj][prop]} than {objects[prop][obj2]}? Answer ONLY Yes or No."},
                ]

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

                A = (obj1 > obj2)
                B = (adj == "pos")
                C = (answer == "yes")
                success = not (not (A ^ B) ^ C)
                
                count += 1
                correct += int(success)
                print(f"{objects[prop][obj1]},{ADJ[adj][prop]},{objects[prop][obj2]},{response},{success}")

                with open(FILENAME, 'a') as f:
                    f.write(f"{objects[prop][obj1]},{ADJ[adj][prop]},{objects[prop][obj2]},{response},{success}\n")

print(f"{correct}, {count}, {correct/count}")