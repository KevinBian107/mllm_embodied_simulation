# UCSD FMP Research
Advancements in computational technology and the diversity of available datasets have led to substantial improvements in Large Language Models (LLMs) and Computer Vision Models (CVMs), leading to the emergence of Multimodal Large Language Models (MLLMs) that integrate together both textual and visual data to improve understanding and interaction capabilities (Dosovitiskiy et al, 2021). Despite these advances, significant gaps remain in our understanding of how MLLMs synthesize and interpret this integrated data, particularly in relation to human-like language comprehension and real-world interaction. The “Black Box” nature of these models further complicates efforts to assess their reliability and interpretability, echoing cognitive linguists’ concerns about symbol grounding issues–how abstract computational entities relate to tangible, real-world entities and experiences (Harnad, 1990).

Embodied simulation, the theory that language comprehension in humans is rooted in physical experiences, offers a framework for evaluating AI’s potential to navigate and understand the world more intuitively. This theory suggests that human language comprehension is profoundly tied to physical experience, a notion supported by cognitive psychologists like Bergen (2015) and Feldman and Narayanan (2003), who posit that such a mechanism is essential for resolving the symbol grounding issue. Empirical support for this theory is evident in phenomena such as the “match effect”, where comprehension in humans is enhanced when sensory experiences align with linguistic inputs (Stanfield & Zwann, 2001; Pecher et al., 2009; Connell, 2007). This hypothesis suggests that when we comprehend language, our brains activate regions similar to those used when we experience the actions or scenarios described. One method to investigate this hypothesis involves examining how humans understand sentences that imply affordance relationships. For instance, comparing the comprehensibility of 'Sam is using a shirt to dry himself after a swim' with 'Sam is using glasses to dry himself after a swim' can reveal insights into this process. 

Given this background, our research turns to machines, specifically Multimodal Large Language Models (MLLMs), to question whether they adopt a similar approach in 'understanding' language. We aim to probe the extent of MLLMs' language comprehension capabilities by testing their ability to perform embodied simulation, particularly through responses to stimuli centered around affordances.

# Clone & Instal
```bash
git clone https://github.com/KevinBian107/fmp_research.git
pip install -r requirements.txt
```

# Running Main Experiment in Study1
```bash
cd study1
python src/main.py --dataset [dataset_name] --model [model_name] --relationship [given relationship]
```
The main.py takes in three command line argument
1. dataset: choose dataset from ['natural','synthetic']
2. model: choose model from ['ViT-B-32','ViT-L-14-336','ViT-H-14','ViT-g-14','ViT-bigG-14','ViT-L-14','imagebind']
3. relationship: relationship is particularly for differentiating the main experiment (afforded v.s. non-afforded) and the follow up manipulation check experiment (related v.s. non-afforded). Choose relationship from ['afforded','related']

# Question & Method
This repository is for studying the potential presence of MLLMs' ability in recognizing affordance realytionships in human languages, hence, showing potential presence of embodied simulation in these multimodal models. *__We are interested in addressing whether multimodal language models can capture the affordances of an object, and its implications in whether artificial systems have some form of embodied cognition.”__*

In our study, we will initially present a textual scenario to various models, accompanied by three images depicting different key objects. Our primary task will be to analyze the probability assigned by the models to each image to determine if their selections align with our hypothesized outcomes.

Given the absence of a pre-existing dataset with images specifically tailored to affordance-related objects, we propose our own unique set of visuals generated using the DALL-E OpenAI model and sourcing relevant images online, each corresponding to the critical objects identified in the previous affordance study by Glenberg & Robertson (2000). 

To ensure the models' accurate recognition of these images, we will conduct a preliminary normalization task. This task, based on the methods of Jones and Trott (2023), will involve a text-to-image matching exercise to verify that each model correctly identifies the images. 

Finally, we will input these images into the models in a structured format—comprising a related image, an image demonstrating the affordance, and a non-afforded image. This setup will be used to evaluate the models' performance in selecting the most appropriate image for each given scenario.

# Models Used
This experiment is conducted using the same MLLMs as was used in the embodied simulation study done by Jones and Trott. This includes six CLIP (developed by OpenAI with various sizes and training dataset) models, one ImageBind Model (developed by Meta), and the latest GPT-4V MLLM:

**For Normalization:**
1. CLIP ViT-B-32
2. CLIP ViT-L-14-336
3. CLIP ViT -L-14
4. CLIP ViT -H-14
5. CLIP ViT -G-14
6. CLIP ViT -bigG-14

**For Experiments:**
1. ImageBind
2. GPT-4V

# Structure of this study
The code is mainly separated into three sections. Each one has its own data format and data frame format.

## Data Collections
Data collection was done both by manual searching online, as well as DALLE generation using OpenAI API. We collected data in 18 triples, one natural set and one synthetic set.

The independent variable is the combination item of text scenarios plus critical object image, which means that each data has the following components:
1. Text scenario
2. Image of affordable condition for critical object
3. Image for non-affordable condition for critical object
4. Image for related condition for critical object

The following is an example of a data point:
Scenario: “After wading barefoot in the lake, Erik needed something to get dry. What would he use?”
1. Relatedobject:[imageofatowel]
2. Affordedobject:[imageofashirt]
3. Non-affordedobject:[imageofglasses]
<img width="500" alt="Screenshot 2023-12-09 at 4 08 33 PM" src="https://github.com/KevinBian107/fmp_research/assets/129793700/e22ae1f8-3032-4540-ad64-e73c2de49105">

We later prompt-engineered 2 methods (explicit, implicit) of asking the model to select the best image for a given scenario to test its understanding of affordance relationships. An example of an explicit version of the prompt vs its implicit version is as follows:
1. Explicit: “What would Erik use to dry himself?”
2. Implicit: “Erik would use this to dry himself”

## Normalizatons of Data
Prior to the main experiment, all data underwent a normalization process. This involved presenting the models with three sets of images (e.g., an image of a towel, glasses, and a shirt) alongside corresponding textual descriptions ("towel", "glasses", "shirt"). The model then engaged in a process of vector space location assignment, utilizing a 3x3 matrix to evaluate matches and mismatches between the images and texts. Following this, we computed the probability for each pair by calculating the dot product of the vector space distances and applying a SoftMax function. This approach allowed us to determine whether the model’s performance on matching pairs aligned with our expectations.

An summary table is produced fro recording the overall data, Random effect model was also performed to ensure equall respondance of different model to the stimulus in normalization.

## Main Experiments
Main experiment can be separated into a few sections
### Study 1 (ImageBind)
1. Creating new data frame
2. Feeding the input into the MLLMs with one condition pairing to two images of the critical images (18 pairs)
3. Each codition has 2 prompts (18 pairs -> 36 pairs)
4. Dot product of vectro space distance and then SoftMax Probability
5. Melting the data frame with id_var as "group_id" and "prompt_type" and value_var being "afforded" and "non_afforded" (36 pairs -> 72 pairs with 36 afforded and 36 non afforded)
6. Clean up in a new data frame
7. Graphing results
8. Statistical testings -> Independent T Test, Two Factor Independent ANOVA

**Notice: When uploaded to github, the imagebind checkpint folder was deleted due to its large size**

### Study 2 (GPT-4V)
Since we do not have access to GPT-4V’s internal embeddings at the time of this study, this test was conducted through interacting with GPT-4V’s API. To obtain a “sensibility ranking” for the use of each object in the images within the context of the 18 scenarios, we issued prompts to the system that were preceded by a specific instruction designed to elicit this ranking:

“In this task, you will read short passages and look at an image of an object. Please rate how sensible it would be to take the action described in the last sentence using the object in
the image in the context of the whole passage. The scale goes from 1 (virtual nonsense) to 7 (completely sensible). Be sure to read the sentences carefully. Please respond only with a number between 1 and 7.”

## Manipulation Check
A follow-up experiment is conducted to compare the effect of "related" images and "non-afforded" images as a manipulation to ensure that fundamental understanding is met.
