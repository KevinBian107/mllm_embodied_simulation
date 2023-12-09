UCSD FMP Research studying the presence of recognizing affordance realytionships in MLLMS

The code is separated into three sections:

1. Data Collections
  Data collection was done both by manual searching online and also DALLE generation using Open AI API
2. Normalizatons of Data
   All data were normalized prior to feeding into the main experiment by providing all the testing models with three images (i.e. [image of an towel], [image of an glasses], [image of an shirt]) and three texts (i.e. "towel", "glasses", "shirt"). Then the model would conduct vector space location assigning with the 3x3 match/mismatch matrix. We then extract such probability using dot product between vector space distance and SoftMax it to retrieve an probability for each of the pairs to see if our expectation of the matching pairs are met.

   All normalization results are formatted into one jupytar notebok
3. Main Experiment (main experiment can be separated into a few sections)
   1. creating new data frame
   2. feeding the input into the MLLMs with one condition pairing to two images of the critical images (18 pairs)
   3. each codition has 2 prompts (18 pairs -> 36 pairs)
   4. dot product of vectro space distance and then SoftMax Probability
   5. melting the data frame with id_var as "group_id" and value_var being "afforded" and "non_afforded" (36 pairs -> 72 pairs with 36 afforded and 36 non afforded)
   6. clean up in a new data frame
   7. graphing results
   8. statistical testings
   
