This is a UCSD FMP Research studying the presence of recognizing affordance realytionships in MLLMS
The code is separated into three sections:
1. Data Collections
  Data collection was done both by manual searching online and also DALLE generation using Open AI API
3. Normalizatons
   All data were normalized prior to feeding into the main experiment by providing all the testing models with three images (i.e. [image of an towel], [image of an glasses], [image of an shirt]) and three texts (i.e. "towel", "glasses", "shirt"). Then the model would conduct vector space location assigning with the 3x3 match/mismatch matrix. We then extract such probability using dot product between vector space distance and SoftMax it to retrieve an probability for each of the pairs to see if our expectation of the matching pairs are met.
5. Main Experiment
