#Thesis Outline



#### Basic TimeLine

#####Baseline prediction
I'm starting out with 'simple' classifiers using a 50/50 test/train split just to see which seem viable for use once we get to iterative screening. This could be a big time suck, because there are now $models^Thyperperamters$ variables to optimize over. 

Models tested so far

* Random Forest
  * Doesn't work stupendously well, we have considered modifying decision threshold to account for class imbalance
* Linear SVM
  * Works pretty well, ~.76 recall on active and inactive.
* RBF SVM
  * Takes forever to train on a 50% train split. Will circle back one we are doing smaller iterative splitting.
* Small NN,
  * 4 layers, relu activations. Works ok, I think that even with focal loss I am hampered by class imbalance
* LightGBM
  * High hopes, haven't tried yet. 

##### Baseline Iterative

Once I establish which models work well in the single run setting, I want to move on to getting a basic iterative process started. This will basically involve decreasing initial training size and then using predictions to select targets for next round of training. Ideally this will yield some soft of minimal viable product in terms of writing up the thesis. There are a few things to think about here. 

* How should I select the next round of targets?  [Maciejewski et al][] Suggest prioritizing compound predicted to be weakly active. But this implies that I either 1) have a confidence measure or 2) start predicting activity levels rather than classifying the compounds.
* Predicting activity? The raw measurements we will be working with are noisy and single replicate. That said, they certainly contain *some* information about how active the compounds are. And by having a continuous measure rather than somewhat arbitrary thresholds (i.e. 3 std above background noise) some learners might perform much better. 

#####Better features

This is critically important, need to find a good way of representing the molecules.Up until this point, I plan to use Morgan fingerprints concatenated with some molecular characteristics. But I'd really like to use graph CNNs to get some other options, [NIPS paper tf implementation is here][]. In particular, the fact that the graph CNNs are full differentiable, means that in theory we can link the learning of embeddings to classifier learning, which might yield some performance increases. 

##### Full RL

Ideally there is time left to explore more exotic approaches to this problem. In particular, I think that reinforcement learning fits our problem really nicely. This [dissertation][] has a good summary of active learning in classification settings (although from above, maybe we don't want to learn classes directly?). 

I specifically think that one-shot learning could be valuable, it basically uses meta/transfer learning to infer properties of a target based on its distance from already labeled examples with the goal being learning to class labels with very few newly labeled examples. There has been [previous work ][]using one-shot learning to predict assay results for a single chemical (i.e. each molecule has data on toxicity, some PK data etc and they are trying to predict on the sparse columns for the dataset). I need to read some more on metric learning, but I think that in theory for multiple screening campaigns with the same compound library, I can use the information about previous activity to inform current predictions (imagine learning a Nx1 vector where N = number of assays available for that library).  Even for different libraries, it may be possible to transfer knowledge based on metrics on the latent embeddings. 

 Additionally, there is a paper on [active one-shot learning][] that combines the one-shot paradigm with a learner that is self aware and able to learn to assess its own uncertainty about its predictions. So in the extreme, we could end up with a learner that has a batch size equal to plate size and can request the contents of the next plate or day of plates in near real time. 

#### Open issues

##### Class imbalance

All of these datasets are wildly imbalanced, usually less than 5% of the data is active. This is skewing all of the classifiers, and seems beyond the simple class weighting corrections that I have tried. Some options to consider are:

* Under/oversampling: Fredrik has previously used clustering (k-means) to aggregate the active and inactive subsets of the data set. Then it is possible to selectively under sample the inactives while still trying to span the embedding space.  Alternatively I can repeatedly sample the actives. 
* Augmenting data. Instead of changing ratios via sampling, I'd prefer to synthesize some new data. There's a new paper from Google about [Unsupervised Data Augmentation][https://arxiv.org/pdf/1904.12848.pdf] that could prove interesting. At the least we could try simple gaussian noise injection.

##### Version Control/ Managing Experimental Progress

I'm aware how quickly I can generate lots of subtle tweaks to the scripts I'm running and lose the previous results stored in RAM as a result.  I think I want to use [MLflow][mlfow.org] to handle storing all the experiments permutations etc. 

##### Compute

Already running into models that exceed my 15 watt i5 (shocker). Once I start playing with graph convolutions and any kind of deep networks, I'm going to need a GPU, and possibly a faster CPU to handle all the RDkit computations if we keep any of that in the embeddings. 

##### Linux

I need to learn how to ssh into linux environs and be comfortable with it. 