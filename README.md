# UCSD FMP Research
Advances in computational capabilities and the diversity of available datasets have led to substantial improvements in the performance of Large Language Models (LLMs) and Computer Vision Models (CVMs). Such progress has generated interest in Multimodal Large Language Models (MLLMs), which integrate textual and visual inputs (Dosovitiskiy et al, 2021). However, much remains unknown about exactly how these models work—in particular, it is not clear how different MLLMs integrate information gleaned from different modalities (e.g., language and vision), and in whether the addition of non-linguistic modalities improves their ability to genuinely “understand” human language. Such a Black Box nature of these models raises concerns about their reliability and interpretability. Current knowledge about how these models represent and interpret the information is minimal, leading to similar issues what cognitive linguists identify as “grounding issues” (Harnad, 1990) in human language comprehension, a challenge concerning the connection of abstract computational representations to real-world meanings.

In human subjects, one potential answer to the “grounding issue” is the embodied simulation hypothesis, which states that during language comprehension, the brain activates similar regions as if the comprehender is “experiencing” it. The ability to comprehend affordance relationships in sentences may be an method in examining the embodied simulation hypothesis in human subjects (i.e. “Sam is using a shirt to dry himself after a swim” is afforded comparing to its counterparts of “Sam is using glasses to dry himself after a swim”). We wonder if machines, particularly MLLMs, would use a similar schematic approach to “understand” language. Thus, we propose to probe MLLMs’ “understanding” of language through testing its ability for embodied simulation through using particularly affordance stimulus.

# Setup
```bash
git clone https://github.com/seantrott/llm_embodiment.git
pip install -r requirements.txt
```

# Question & Method
This repository is for studying the potential presence of MLLMs' ability in recognizing affordance realytionships in human languages, hence, showing potential presence of embodied simulation in these multimodal models. *__We are interested in addressing the question of “Do multimodal language models capture the affordance relationships of objects in languages and, thus, showing evidence of embodied simulation?”__*

In summary, we will be presenting a scenario in text to the models and then provide 3 images of the 3 different critical objects for the models to choose. Then, we would extract the probability assigned to each of the images and see if they would match our expectations.

Due to lack of existing data set on image data on affordance critical object, we propose to both generate images from the DALLE Open AI model and collect images online for each of the according critical object in the previous affordance study from (Glenberg & Robertson, 2000). Afterwards, a normalization task would be performed with all the testing MLLMs from Jones and Trott (2023) to ensure that they “understand” all these images separately by doing an initial text/image matching. At last, all of the images would be feed into the testing MLLMs in a triple pair format (image_related x image_afforded x image_non-afforded) for selection after prompting the model with a particular scenario.

# Models Used
The experiment is conducted using the same MLLMs as Jones and Trott did in their study into MLLMs’ embodied simulation, which includes six CLIP (developed by Open AI with various sizes and various training data set) model and one ImageBind Model (developed by Meta). Particularly, we are using the models listed below:
1. CLIP ViT-B-32
2. CLIP ViT-L-14-336
3. CLIP ViT -L-14
4. CLIP ViT -H-14
5. CLIP ViT -G-14
6. CLIP ViT -bigG-14
7. ImageBind

# Structure of this study
The code is mainly separated into three sections & each one has its own data format and data frame format:

## Data Collections
Data collection was done both by manual searching online and also DALLE generation using Open AI API. We collected 18 triple pairs of data, which constitutes our independent variable

The independent variable is the combination item of text scenarios plus critical object image, whcih means that each data has the following components:
1. Text scenario
2. Image of affordable condition for critical object
3. Image for non-affordable condition for critical object
4. Image fro related condition for critical object

The following is an example of a data point:
Scenario: “After wading barefoot in the lake, Erik needed something to get dry. What would he use?”
1. Relatedobject:[imageofatowel]
2. Affordedobject:[imageofashirt]
3. Non-affordedobject:[imageofglasses]
<img width="317" alt="Screenshot 2023-12-09 at 4 08 33 PM" src="https://github.com/KevinBian107/fmp_research/assets/129793700/e22ae1f8-3032-4540-ad64-e73c2de49105">

We later on prompt engineered 2 ways (explicit, implicit) of asking the model to select the best image for an given scenario to test its understanding of affordance realtionships, which boost 18 data points to 32 data points
1. Explicit: “What would Erik use to dry himself?”
2. Implicit: “Erik would use this to dry himself”

## Normalizatons of Data
All data were normalized prior to feeding into the main experiment by providing all the testing models with three images (i.e. [image of an towel], [image of an glasses], [image of an shirt]) and three texts (i.e. "towel", "glasses", "shirt"). Then the model would conduct vector space location assigning with the 3x3 match/mismatch matrix. We then extract such probability using dot product between vector space distance and SoftMax it to retrieve an probability for each of the pairs to see if our expectation of the matching pairs are met.

All normalization results are formatted into one jupytar notebok

## Main Experiment (main experiment can be separated into a few sections)
1. Creating new data frame
2. Feeding the input into the MLLMs with one condition pairing to two images of the critical images (18 pairs)
3. Each codition has 2 prompts (18 pairs -> 36 pairs)
4. Dot product of vectro space distance and then SoftMax Probability
5. Melting the data frame with id_var as "group_id" and "prompt_type" and value_var being "afforded" and "non_afforded" (36 pairs -> 72 pairs with 36 afforded and 36 non afforded)
6. Clean up in a new data frame
7. Graphing results
8. Statistical testings -> Independent T Test, Two Factor Independent ANOVA
   
