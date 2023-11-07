# -*- coding: utf-8 -*-
import os
import argparse
import warnings


# Suppress specific warnings
warnings.filterwarnings("ignore", module="torchvision.transforms._functional_video")
warnings.filterwarnings("ignore", module="torchvision.transforms._transforms_video")

import torch
import open_clip
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from utils import (setup_model, format_results,
                   results_summary, ttest, plot_results)


def format_sentences_color(obj, image_paths):
    color1 = image_paths[0].split(".")[0].split()[-1]
    color2 = image_paths[1].split(".")[0].split()[-1]
    return ['It is a {color} {x}'.format(color = color1, x = obj),
           'It is a {color} {x}'.format(color = color2, x=obj)]

def format_sentences_orientation(obj):
    return ['It is a horizontal {x}'.format(x = obj),
           'It is a vertical {x}'.format(x=obj)]

def analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder, dataset):
    """New analyze_data function for manipulation check."""
    df = pd.read_csv(csv_path)
    all_results = []

    for index, item in tqdm(df.iterrows(), total=len(df)):
            
        text_list = [
                        item['sentence_a'].strip(), item['sentence_b'].strip()]

        
        image_paths = [os.path.join(img_folder, item['image_a']),
                       os.path.join(img_folder, item['image_b'])]

        ### Assemble sentences 
        ## TODO: Do for orientation too
        if dataset == "connell2007":
            text_list = format_sentences_color(item['object'], image_paths)
        elif dataset == "muraki2021":
            text_list = format_sentences_orientation(item['object'])
        else: 
            raise ValueError("No protocol defined for {d}".format(d = dataset))
        # image_paths = [path.replace(".jpg", ".png") for path in image_paths]
        
        if isinstance(model, open_clip.model.CLIP):

            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            text_inputs = tokenizer(text_list)



            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                image_features = torch.stack([model.encode_image(img_input) for img_input in image_inputs]).squeeze()
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
                # Calculate the similarity
                results = torch.softmax(
                    text_features @ image_features.T, dim=-1)

        elif isinstance(model, clip.model.CLIP):

            text_inputs = clip.tokenize(text_list).to(device)
            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                image_features = [model.encode_image(img_input) for img_input in image_inputs]
                # Calculate the similarity
                results = torch.softmax(
                    text_features @ torch.stack(image_features).squeeze().T, dim=-1)

        elif isinstance(model, imagebind_model.ImageBindModel):

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
                ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            }

            with torch.no_grad():
                embeddings = model(inputs)

            results = torch.softmax(
                embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)
            
        else:
            raise ValueError("Model must be either 'clip' or 'imagebind'")

        all_results.append({
            'match_a': results[0][0].item(),
            'mismatch_a': results[0][1].item(),
            'match_b': results[1][1].item(),
            'mismatch_b': results[1][0].item(),
            'object': item['object']
        })

    return pd.DataFrame(all_results)

def main(args):

    # Parse arguments
    model_name = args.model
    dataset = args.dataset

    # Set up paths
    model, preprocess, tokenizer, device = setup_model(model_name)
    csv_path = f"data/{dataset}/items.csv"
    img_folder = f"data/{dataset}/images"
    img_save_path = f"results/{dataset}/{dataset}_{model_name}_mc.png"
    data_save_path = f"results/{dataset}/{dataset}_{model_name}_mc.csv"

    # Create folders for results
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # Run analysis
    results_raw = analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder, dataset)
    results = format_results(results_raw, model_name, dataset)
    summary = results_summary(results)

    # Print and save results
    t, p = ttest(results)
    print(summary)
    print(f"t = {t}, p = {p}")
    plot_results(results, img_save_path)
    results.to_csv(data_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LLM datasets.')
    parser.add_argument('--dataset', type=str, required=True, choices=['connell2007', 'muraki2021', 'pecher2006'],
                        help='Name of the dataset to process')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use for analysis')
    args = parser.parse_args()
    main(args)