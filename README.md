### Python implementation for the ADReSS Challenge (2020) (classification task) ###

**Used data:**
- DementiaBank - Pitt Corpus (selected audios from the Cookie theft picture description task).
- See: https://dementia.talkbank.org/ADReSS-2020/ 

**Used features:**
- Computed functional level features of a compare set using openSMILE library for each audio (in total, 6373 features for one audio recording). 
- See: https://audeering.github.io/opensmile-python/ 

**Used classifier:** 
- Multilayer perceptron model (MLP) consisting of ReLU units in a hidden layer and soft-max activation function in an output layer. 


