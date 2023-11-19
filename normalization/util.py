import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import clip
import open_clip

def preprocess_image(input_folder, output_folder, target_size=(224,224), padding_color=(255,255,255)):
    """
    Preprocesses images:
    1. Resizing them to a target size
    2. Padding them if it is necessary
    3. Saves the processed images in the output folder
    
    Input
        input_folder (str): Path to the folder containing original images
        output_folder (str): Path to the folder to save preprocessed images
        target_size (int, optional): Target size for the image (Default is 224)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more formats if needed
            image_path = os.path.join(input_folder, filename)

            with Image.open(image_path) as img:
                
                # Convert to RGB if necessary
                if img.mode == 'RGBA':
                    img = Image.alpha_composite(Image.new("RGBA", img.size, padding_color), img)
                    img = img.convert("RGB")
                    
                img_aspect = img.width / img.height
                target_aspect = target_size[0] / target_size[1]

                # Resize image
                if img_aspect > target_aspect:
                    new_width = target_size[0]
                    new_height = int(target_size[0] / img_aspect)
                else:
                    new_height = target_size[1]
                    new_width = int(target_size[1] * img_aspect)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Pad image
                left_padding = (target_size[0] - new_width) // 2
                right_padding = target_size[0] - new_width - left_padding
                top_padding = (target_size[1] - new_height) // 2
                bottom_padding = target_size[1] - new_height - top_padding

                padded_img = Image.new("RGB", target_size, color=(255,255,255))
                padded_img.paste(img, (left_padding, top_padding))

                # Save the preprocessed image
                filename = filename.split('.')[0] + '.png'
                save_path = os.path.join(output_folder, filename)
                padded_img.save(save_path)

def expand2square(pil_img, background_color=(255,255,255)):
    '''
    1. Expanding to a square images
    2. Make background color standard
    '''
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result