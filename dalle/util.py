import os
import json
from pathlib import Path
from openai import OpenAI
from base64 import b64decode

def generate(prompt_input):
    '''
    1. Takes i nan prompt and return the images generated to an specified folder
    2. Returns file name
    '''
    #Path for savings
    DATA_DIR = Path.cwd() / "responses"
    DATA_DIR.mkdir(exist_ok=True)

    #Create DALLE Object
    client = OpenAI()

    #Call for generation using DALLE
    response = client.images.generate(
        prompt=prompt_input,
        n=1,
        size="256x256",
        response_format="b64_json")

    #compact generated responses into Json Files
    file_name = DATA_DIR / f"{prompt_input[:5]}-{response.created}.json"

    with open(file_name, mode="w", encoding="utf-8") as file:
        json.dump(response.data[0].b64_json, file)
    #print(response.data[0].b64_json)

    return file_name

def convert(file_name):
    '''
    1. Takes in file_name string
    2. Saved a PNG image as a Base64-encoded string in a JSON file. Now convert back to PNG
    '''
    JSON_FILE = file_name
    IMAGE_DIR = Path.cwd() / "images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(JSON_FILE, mode="r", encoding="utf-8") as file:
        response = json.load(file)

    for index, image_dict in enumerate(response):
       image_data = b64decode(image_dict["b64_json"])
       image_file = IMAGE_DIR / f"{JSON_FILE.stem}-{index}.png"
       with open(image_file, mode="wb") as png:
          png.write(image_data)
