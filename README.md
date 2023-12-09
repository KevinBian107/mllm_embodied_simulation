# UCSD FMP Research
## Introduction
This repository is for studying the potential presence of MLLMs' ability in recognizing affordance realytionships in human languages, hence, showing potential presence of embodied simulation in these multimodal models.

In summary, we will be presenting a scenario in text to the models and then provide 3 images of the 3 different critical objects for the models to choose. Then, we would extract the probability assigned to each of the images and see if they would match our expectations.

Due to lack of existing data set on image data on affordance critical object, we propose to both generate images from the DALLE Open AI model and collect images online for each of the according critical object in the previous affordance study from (Glenberg & Robertson, 2000). Afterwards, a normalization task would be performed with all the testing MLLMs from Jones and Trott (2023) to ensure that they “understand” all these images separately by doing an initial text/image matching. At last, all of the images would be feed into the testing MLLMs in a triple pair format (image_related x image_afforded x image_non-afforded) for selection after prompting the model with a particular scenario.
The code is mainly separated into three sections & each one has its own data format and data frame format:

## Data Collections
Data collection was done both by manual searching online and also DALLE generation using Open AI API


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
   
