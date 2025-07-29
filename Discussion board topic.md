### Notes on semantics:
Class: class here refers to the 'data science' class, ie, the classification of the organism, and not the taxonomic level 'class'. ie the class could be the taxonomic order 'chordata' or the genus 'octopus' or the species Octypus rubescens' etc. 

### Data breakdown:
The data is arranged into whole images, which are further annotated with bounding boxes (ROIs) for each individual organism of interest in the images. The data also comes with a taxonomic hierarchy which dictates the loss.

### Dataset Split:
Since there are multiple ROIs per image, and many of the ROIs have very similar illumination and backgrounds if they are in the same image, it seemed best to split the dataset into a training and validation set based on the whole images. This also helps since the test ROIs are from new (unseen) images. 

The images chosen for the training and validation split are in the csv file on the github.

## Problem Overview
This is a hierarchical classification problem, where we are interested in classifying each organism, and we are penalized by the distance between our predicted class and the expert determined class based on the distance along the taxonomic tree. This incentivizes us to be close to the predicted class.



## Generating a Distance Matrix from the Tree
An important step to including the losses along the tree is to convert the tree into a distance/loss matrix which we could then use to compute the 'actual' training and validation loss. 

The full notebook to make a distance matrix from a tree can be found here:

For the purposes of illustration, let's say we have the simplified tree below:
```
              |
           Chordata (Crustaceans)
     _________|__________
    |                    | 1
    |              Actinopterygii
    | 5          ________|________
    |           |                 |
    |           |                 |
 Pyrosom        | 4               | 4
 atlanticum     |                 |
             Sebastes         Merluccius
             diploproa        productus
```
Let's say we take the two highlighted classes (Pyrosoma atlanticum and Actinopterygii), and calculate the shortest distance along the tree between these two species. Then we get 5+1 = x.

Now we can add this to our distance matrix:
```
          Chor. | Acti. | P. a. | S. d. | M. p.
Chor.           |       |       |       |      
Acti.           |       |   6   |       |
P. a.           |   6   |       |       |
S. d.           |       |       |       |
M. p.           |       |       |       |
```


We do this for all possible classifications, and finally we have a distance / loss per predicted/true class pair, as follows:
```
      Chor. | Acti. | P. a. | S. d. | M. p.
Chor.   0   |   1   |   5   |   5   |   5   
Acti.   1   |   0   |   6   |   4   |   4
P. a.   5   |   6   |   0   |  10   |  10
S. d.   5   |   4   |  10   |   0   |   8
M. p.   5   |   4   |  10   |   8   |   0
```

This is handy so we do not need to continuously query the FathomNet API each time we want to compute the expected loss. We can instead look up the loss in this table. We can also now easily compute the actual loss 

## Choosing the Class - Maximum Probability vs Minimum Expected Loss

### Interlude - How to think about each set and subset (ie genus and species set)
Before we get into choosing the class, I want to briefly note one thing that will be important to this section. The typical way we would think about the predicting that an roi belongs to the given hierarchical class (ie Actinopterygii) is that each set includes all of the sub sets. Ie, if you predicit an ROI as 'Mercullius productus' with probability 60%, the probability of it being in the class 'Actinopterygii' should be at least 60%, since all Mercullius productus are also members of Actinopterygii.

Instead, I decided to think of each class as the 'leftovers' of the subsets. Ie, if the algorithm thought it had a 60% chance of being 'Mercullius productus', but a 5% chance of being another kind of Mercullius, then the class representing 'Mercullius' would be given a probability 5%. This allowed me to still use softmax loss on the final layer, as the probability of all 79 classes summed together would therefore sum to 1, and I could treat it as a probability. 

### Maximum Probability vs Minimum Expected Loss
One thing that consistently worked well in this challenge, was to choose the class which minimized the overall expected loss based on the probabilities the model outputted. 

Normally, we would simply predict the class which has the highest probability. Called Maximum Probability (MP). However, since we have this nice distance matrix, and probabilities of each class, we could instead choose the class which minimizes the overall expected loss based on the distance matrix. 

To demonstrate this, if we again take our distance matrix, and we now have a vector of probabilities for each class outputted by our model. 
```
            Distance Matrix                            Probabilities of each class
            
      Chor. | Acti. | P. a. | S. d. | M. p.                   Prob.
Chor.   0   |   1   |   5   |   5   |   5             Chor.   0.02
Acti.   1   |   0   |   6   |   4   |   4             Acti.   0.14
P. a.   5   |   6   |   0   |  10   |  10             P. a.   0.38
S. d.   5   |   4   |  10   |   0   |   8             S. d.   0.04
M. p.   5   |   4   |  10   |   8   |   0             M. p.   0.52
```

Given these probabilities, we can see that Merluccius productus is predicted as the most likely class with probability 0.52, but Pyrosoma atlanticum is nearly as close with probability 0.38, and these two classes have a high distance from eachother on the tree (10). So instead it might be advantageous to choose a class which is 'between' these two classes in the tree. But how to choose?

Given these probabilities, we can calculate the expected loss if we predict each class. Starting with the class with the highest probability, Merluccius productus. The expected loss for this class is E(loss) = sum(loss*p(loss)) = 5 * 0.02 + 4 * 0.14 + 10 * 0.38 + 8 * 0.04 + 0 * 0.52. This is essentially the dot product between the column in the distance matrix and the vector of probabilities. 

If we do this calculation for each class, we get the following: 
```
       Expected Loss
Chor.   4.84
Acti.   4.54
P. a.   6.54
S. d.   8.62
M. p.   4.78
```

This is what we expect the loss to be if we choose this class, based on the probabilities of each class, and the distance matrix given by the tree.

From this we can see that, Actinopterygii, despite having a low probability of being that class, has a lower overall expected loss given that it is between the different classes.

This can easily be represented as a matrix multiplication between the distance matrix and the probability vector. We can then use an argmin function to find the class with the lowest expected loss, and choose this as our predicted class. 

## Graphical Neural Networks
I noticed that in many cases there were multiple instances of the same organism in a single image, and in many cases, such as with this large group of the same species below, many of the individual organisms were blurred, but still likely to be the same class as the sharper organisms, as they were clearly in proximity. To account for this, I tried to implement a graphical neural network layer which would generate a graph from other ROIs in the same image, and hopefully use those to predict the class better. 

### Interlude - Sampling issues
I however ran into the issue that my batch sizes were quite small (40) and it was highly unlikely that there would be multiple ROIs from the same image in each batch. Therefore I implemented a new sampler which would sample a random number of ROIs from each image, so I was almost guaranteed that each ROI would have 'sibling' ROIs from the same image in each batch, but there would still be some randomness in which 'siblings' would appear to reduce the chance of overfitting. 

### Onto the GNN
With the sampling solved, I could then given a batch, generate a fully interconnected graph of all ROIs that came from the same image within that batch. I could then feed this into a GNN layer, and feed that output to a final classification layer, alongside the pure output from the linear layer. 


## Final Tweaks
The final best model I had also used hierarchical triplet loss with a weight of 0.001 and hierarchical cross entropy (HXE). At the time I was not using seeds properly, so I have not reproduced the exact results of the model. However, the model is available on hugging face https://huggingface.co/FathomNet/FathomNet-2025-5thplace-Model 

## Failures
Here are some things that didn't really work:
1. Using the distance directly in the loss function (ie trying to minimize the distance matrix * probabilities)
2. Setting up the tree covariance matrix as the loss function instead
3. Various tweaks to the graphical neural network

## Things I wish I had done instead:
1. Setting the seed earlier
2. Focussing on making the data and model code cleaner 
3. Played more with ensembles
4. The implementation of the GNN didn't completely seem to solve these issues when there was a school of fish. Perhaps there is a better way to implement this in a graph.

## Some results (Best results starred **)

Notes on results:
1. EfficientNet: b0 was used as the base for all the models. 
2. Data Aug means adding Cutmix and Mixup data augmentations. 
3. EM means adding expected loss minimization as explained above. 
4. GNN means adding a Graph Neural Network layer
5. HXE means adding a hierarchical cross entropy loss
6. Triplet means adding a hierarchical triplet loss
7. Each result is only one run. All results used seed 0, except the best model
8. Note that the validation loss is quite different from the test set loss. See https://www.kaggle.com/competitions/fathomnet-2025/discussion/573371 for more details. I suspect this is due to the test set having a different distribution than the published training set. 
```
                                                                 validation | public test set | private test set
Baseline EfficientNet:                                             0.9216   |      3.10       |       2.72
EfficientNet + Data Aug:                                           0.8094   |      2.94       |       2.68
EfficientNet + Data Aug + EM:                                      0.7649   |      2.98       |       2.61        
EfficientNet + Data Aug + GNN:                                     0.6397   |      2.83       |       2.63      
EfficientNet + Data Aug + GNN + EM:                               *0.5937*  |      2.75       |       2.39               
EfficientNet + Data Aug + HXE:                                     0.7609   |      2.95       |       2.60          
EfficientNet + Data Aug + HXE + EM:                                0.7326   |      2.95       |       2.60     
Best Model (EfficientNet + DataAug + EM + HXE + Triplet + GNN):    0.6381   |     *2.50*      |      *2.19*
```

## Links
Competition: https://www.kaggle.com/competitions/fathomnet-2025/overview

Github: https://github.com/robertahunt/FathomNet2025 

Hugging Face with 5th place model: https://huggingface.co/FathomNet/FathomNet-2025-5thplace-Model