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

def setup_model(model_name):
    '''
    Checking models
    '''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if model_name == "ViT-B-32":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')

    elif model_name == "ViT-L-14-336":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')

    elif model_name == "ViT-H-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k')

    elif model_name == "ViT-g-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b88k')

    elif model_name == "ViT-bigG-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s39b_b160k')

    elif model_name == "ViT-L-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    
    elif model_name == "imagebind":
        model = imagebind_model.imagebind_huge(pretrained=True)
        preprocess = None

    else:
        raise ValueError("Model not implemented")
    
    if isinstance(model, open_clip.model.CLIP):
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        tokenizer = None
    
    model.eval()
    model.to(device)
    return model, preprocess, tokenizer, device

def analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder, relationship):
    '''
    1. Reads in an data frame
    2. Extract texts and images for using model
    3. Feeding into different model depedns on isnatcnes
    '''
    #Labeled data set passed in
    df = pd.read_csv(csv_path)
    all_results = []

    for index, item in tqdm(df.iterrows(), total=len(df)):
        # afforded vs non-afforded
        #retrieve separate information for text & images
        text_list = [item['condition'].strip()] ### only a single text string
        image_paths = [os.path.join(img_folder, item[relationship+'_image']),
                       os.path.join(img_folder, item['non-afforded_image'])]
        
        #model 1 Open AI CLIP
        if isinstance(model, open_clip.model.CLIP):
            
            #tokenize & preprocess images
            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            text_inputs = tokenizer(text_list)
            
            with torch.no_grad(): #no tracking of gradient for space efficiency
                text_features = model.encode_text(text_inputs)
                image_features = torch.stack([model.encode_image(img_input) for img_input in image_inputs]).squeeze()
                
                # Calculate Similarity (words to images)
                results = torch.softmax(text_features @ image_features.T, dim=-1)

        #model 2 Meta Image Bind
        elif isinstance(model, imagebind_model.ImageBindModel):
            
            #tokenize & preprocess images
            inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device),
                      ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
            
            with torch.no_grad():
                embeddings = model(inputs)

                # Calculate Similarity (words to images)
                results = torch.softmax(embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)
            
        else:
            raise ValueError("Model must be either 'clip' or 'imagebind'")

        #Put everything in a table
        all_results.append({
            ### replace with probability for afforded and non-afforded
            relationship: results[0][0].item(),
            'non_afforded': results[0][1].item(),

            'prompt_type': item['prompt_type'],
            'group_id': item['group_id']

        })

    return pd.DataFrame(all_results)

def format_results(df, model_name, dataset, relationship):
    '''
    Melting & reformatting the result
    Melting is essentially making some of the columns in the data frame as a tag for variable, making wide df to long df

    1. Select id adn prompt type to be the id, afforded and non afforded as the variables
    2. 36 conditions, 72 separate afforded and non afforded conditions

    '''
    melted_df = pd.melt(df, id_vars = ["group_id",'prompt_type'], value_vars=[relationship, 'non_afforded'])

    #Rename columns
    melted_df['relationships'] = melted_df['variable']
    melted_df = melted_df.rename(columns={'value': 'probability'}).drop(columns=['variable'])
    print(melted_df)

    #Formatting
    melted_df = melted_df[["relationships", 'prompt_type', "probability", "group_id"]]
    melted_df["model"] = model_name
    melted_df["dataset"] = dataset
    return melted_df

def results_summary(df):
    '''
    Producing summary for data frame given in
    '''
    summary = df[["relationships", "probability"]].groupby(["relationships"]).mean()
    return summary

def ttest(df, relationship):
    '''
    Conducting Independnet T Test
    '''
    from scipy.stats import ttest_ind
    other_relationship = df[df["relationships"] == relationship]["probability"]
    non_afforded = df[df["relationships"] == "non_afforded"]["probability"]
    t, p_t = ttest_ind(other_relationship, non_afforded)

    return t, p_t

def anova(df):
    '''
    Perforem Two Factor ANOVA
    '''
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols
  
    # Performing two-way ANOVA 
    ### 'probability ~ relationships + prompt_type + relationship:prompt_type'
    model = ols('probability ~ C(relationships) + C(prompt_type) + C(relationships):C(prompt_type)', data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2) 

    return anova_result

def plot_results(df, save_path=None):
    '''
    Plot results and save plots
    '''
    sns.barplot(data=df, x="relationships", y="probability", hue = "prompt_type")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()