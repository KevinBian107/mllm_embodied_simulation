# -*- coding: utf-8 -*-
import os
import argparse
import warnings
from util import (setup_model, analyze_data, format_results, results_summary, ttest, plot_results)


# Suppress specific warnings
warnings.filterwarnings("ignore", module="torchvision.transforms._functional_video")
warnings.filterwarnings("ignore", module="torchvision.transforms._transforms_video")

def main(args):

    # Parse arguments -> terminal passing in directly
    model_name = args.model
    dataset = args.dataset

    # Set up paths
    model, preprocess, tokenizer, device = setup_model(model_name)
    csv_path = f"data/normalization.csv"

    img_natural_folder = f"data/{dataset}/images_natural"
    img_synthetic_folder = f"data/{dataset}/images_synthetic"

    img_natural_save_path = f"results/natural/{dataset}/{dataset}_{model_name}.png"
    data_natural_save_path = f"results/natural/{dataset}/{dataset}_{model_name}.csv"

    img_synthetic_save_path = f"results/synthetic/{dataset}/{dataset}_{model_name}.png"
    data_synthetic_save_path = f"results/synthetic/{dataset}/{dataset}_{model_name}.csv"

    # Create folders for results
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # Run analysis
    natural_results_raw = analyze_data(model, preprocess, tokenizer, device, csv_path, img_natural_folder)
    synthetic_results_raw = analyze_data(model, preprocess, tokenizer, device, csv_path, img_synthetic_folder)
    natural_results = format_results(natural_results_raw, model_name, dataset)
    synthetic_results = format_results(synthetic_results_raw, model_name, dataset)
    natural_summary = results_summary(natural_results)
    synthetic_summary = results_summary(synthetic_results)


    # Print and save results for natural data
    t, p = ttest(natural_results)
    print(natural_summary)
    print(f"t = {t}, p = {p}")
    plot_results(natural_results, img_natural_save_path)
    natural_results.to_csv(data_natural_save_path)

    # Print and save results for synthetic data
    t, p = ttest(synthetic_results)
    print(synthetic_summary)
    print(f"t = {t}, p = {p}")
    plot_results(synthetic_results, img_synthetic_save_path)
    synthetic_results.to_csv(data_synthetic_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LLM datasets.')
    parser.add_argument('--dataset', type=str, required=True, choices=['connell2007', 'muraki2021', 'pecher2006'],
                        help='Name of the dataset to process')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use for analysis')
    args = parser.parse_args()
    main(args)