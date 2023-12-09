# -*- coding: utf-8 -*-
import os
import argparse
import warnings
from util import (setup_model, analyze_data, format_results, results_summary, ttest, plot_results)


# Suppress specific warnings
warnings.filterwarnings("ignore", module="torchvision.transforms._functional_video")
warnings.filterwarnings("ignore", module="torchvision.transforms._transforms_video")

def main(args):
    '''
    Main function to call on, select model and dataset, call on args
    '''

    # Parse arguments -> terminal passing in directly
    model_name = args.model
    dataset = args.dataset

    # Set up paths
    model, preprocess, tokenizer, device = setup_model(model_name)
    
    csv_path = f"data/{dataset}/affordance_{dataset}.csv"
    img_folder = f"data/{dataset}/images"

    img_save_path = f"results/{dataset}/{dataset}_{model_name}.png"
    data_save_path = f"results/{dataset}/{dataset}_{model_name}.csv"


    # Create folders for results
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # Run analysis
    results_raw = analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder)
    results = format_results(results_raw, model_name, dataset)
    summary = results_summary(results)

    # Print and save results for data slected
    t, p = ttest(results)
    print(summary)
    print(f"t = {t}, p = {p}")
    plot_results(results, img_save_path)
    results.to_csv(data_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LLM datasets.')
    parser.add_argument('--dataset', type=str, required=True, choices=['natural','synthetic'],
                        help='Name of the dataset to process')
    parser.add_argument('--model', type=str, required=True, choices = ['ViT-B-32','ViT-L-14-336','ViT-H-14','ViT-g-14','ViT-bigG-14','ViT-L-14','imagebind'],
                        help='Model to use for analysis')
    args = parser.parse_args()
    main(args)