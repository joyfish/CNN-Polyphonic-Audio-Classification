# CNN-Polyphonic-Audio-Classification
A CNN based classification architecture, built to handle simultaneous sound events.

Within this project there are files for general audio augmentation, preprocessing, sound mixing, clustering, blind source separation, and different techniques for classifying both single and mixed sound events using convolutional neural networks.

The CNN architecture used was the VGGish, with the final model (CNN_mixed_binary2.py) supplementing this with binary classifiers for each class, with the results feeding into a stacked ANN.

The folder titled audioset, should be merged with that of the VGGish project, found at: "https://github.com/tensorflow/models/tree/master/research/audioset"

Any files made by the VGGish creaters that are used here, show their copyright at the top, and all rights go to them. The additional files are what have been created for this project, which uses VGGish with additional techniques to try and robustly classify mixed events. 
