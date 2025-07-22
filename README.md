# FathomNet 2025 Kaggle Competition Code

This github contains the code for the FathomNet 2025 kaggle competition which achieved 5th place overall. 

Competition Link: https://www.kaggle.com/competitions/fathomnet-2025
Link to 5th place checkpoint: https://huggingface.co/FathomNet/FathomNet-2025-5thplace-Model/tree/main

Things I think worked well:
1. Choosing the class which **minimized the expected loss based on the distance matrix (and not the class with the highest probability)** and implementing this as a matrix multiplication with the distance matrix.
2. Implementing a small Graph Neural Network layer - the idea here was to help in cases where there are many specimens of the same species in a single overall image, and one is easy to classify, but the other instances may be blurry. Then adding a graph layer could help guide the model to the correct classification.
3. Using EfficientNet as a simple and fast base network, made experimenting fairly fast and simple.

Things I wish I had done differently:
1. Setting the seed earlier: I initially used a random seed each time, which made measuring progress and reproducing results difficult. It wasn't until near the end I changed this. This is why the results using this seed also sadly do not match perfectly with the public results.
2. Making the saving and logging process cleaner overall so it would be easier to compare results.
