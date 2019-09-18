# Thesis Plan Outline

### Gabriel Dreiman MSc Machine Learning



The thesis revolves around high throughput screening (hence HTS) and trying to reduce the number of screens required to find the active compounds in a drug library. In general HTS is expensive and has a relatively low yield (often well below 5%). The high cost and low yield make it difficult to run exploratory campaigns, thus reducing the former and increasing the later would allow for increased exploration of drug libraries. I intend to use an iterative screening/active learning approach to try to achieve these increases in efficiency and yield.  In essence this approach breaks the library into a series of subsets which are screened one at a time. After each screening, an algorithm (hence agent) trains on the screened data, and (once trained) specifies which members of the untested pool should be included in the next screening run. Ideally the agent learns the characteristics of active compounds, and can prioritize their screening. In doing so, costs are reduced, and yields increased.  The path to this final active learning agent is established below, split into two steps. In the first, I aim to establish a performance hierarchy for various methods of predicting activity in a non-iterative setting. In the second, I expand the best approaches to the iterative paradigm. The first step is split into two related sub-questions. The first has to do with establishing the optimal agent, namely it deals with determining which machine learning algorithm performs best on a test data set. The second, which must be answered in parallel, addresses how best to represent the molecules that compose the library. This second question is of more interest, as the method of representing the molecules will have an outsize impact on the performance of the classifiers. 

## 0) Acquire Data

Initial Data is sourced from PubChem BioAssays. I've intentionally selected HTS campaigns with between 50,000 and 200,000 molecules in their libraries. I'm considering intentionally selecting multiple datasets that use the same library as some studies suggest that there is merit to transfer learning in [drug classification](<https://arxiv.org/abs/1502.02072>). 	
| AID         | Active/Total | Description                                                | Active_precent |
| ----------- | ------------ | ---------------------------------------------------------- | -------------- |
| AID_1345083 | 6153/93211   | Phenotypic for Cytotoxicity on HEK 293                     | 6.6            |
| AID_624255  | 4582/76537   | Luminecent Phenotypic for T.cruzi proliferation inhibition | 6.0            |
| AID_449739  | 4230/104728  | Inhibitors of Cav3 T-type Calcium Channels                 | 4.0            |
| AID_995     | 707/70898    | ERK Inhibition                                             | 1.0            |
| AID_938     | 1794/72026   | Thyroid Stimulating Hormone Receptor Agonists              | 2.5            |
| AID_628     | 2179/63656   |                                                            | 3.4            |
| AID_596     | 1391/69668   |                                                            | 2.0            |
| AID_893     | 5649/73912   |                                                            | 7.6            |
| AID_894     | 6428/148481  |                                                            | 4.3            |
|             |              |                                                            |                |

### GHD Comments

> I dropped one of the AIDs (605) because it had like 35 actives and thats just silly
>
> AID_596, AID_995, AID_938 all come from roughly same library, and they are cell based assays so that aligns more with our possible results. 

## 1) Benchmark Classifiers 

### A) Compare Classifiers 

​	The goal here is to establish which Machine Learning (ML) algorithms are best suited to the task of predicting activity of molecules against a target compound. An important nuance: HTS actually measures a  raw biological signal to determine molecule-protein affinity. This signal is relatively noisy, due to background noise, non-specific binding etc, and usually only a single replicate is performed. Thus many datasets establish a threshold (3 stdev above the expected background noise for example) and label all compounds with signals above this as 'Active' and those falling below as 'Inactive' or 'Inconclusive'. Currently my plan is to treat these labels as classes and train classification agents to predict activity. However, there is an argument to be made for either a) changing the classes to binned fluorescence signals (i.e. 30, 40, 50 etc) or b) training agents as regressors trying to fit the embedding to the signal. I think that the raw signal contains more information than simple classes (i.e. knowing that a compound just misses the cutoff for activity may signal merit in exploring embedding space around it), but it might be hard to fit to what is expected to be a very noisy signal. 

It is a bit complicated to extract meaningful information about iterative performance from experiments that only train once on a fixed set of data. However, iterative training is very computationally expensive, so I want to discard futile approaches as soon as possible. To gain some insight into iterative performance, I will be using a bifurcated cross folding approach to benchmarking classifier performance. After reserving a 20% validation split, I intend to train each agent both on the full 80% Train split and to do 8-fold cross validation on 8 folds of the full Train split. Hopefully the performance on smaller data sets will replicate the expected performance from the agent for the first iteration in the iterative approach while the agent trained on the full dataset will inform about the agent's performance in later iterations.

> Initially ran experiments with all classifiers on the MFP and Mol Chars embeddings. Ran it both on a 10% selection and an 80% selection. At this point it seemed like SVMs were the best, but it turns out that they're basically just over optimistic.

### B) Compare Embeddings

The are two canonical approaches to transforming the chemical structure of a compound into a machine-interpretable format that agents can learn from. The most basic is the extraction of a vector containing information about the chemical-physical properties of the molecule. This vector in continuous and can capture key characteristics for drug discovery (hydrophobicity for example). However, [Although simple to compute, 1D descriptors suffer from degeneracy problems where distinct compounds are mapped to identical descriptor values for a given descriptor. ][Molecular Char Degeneracy] The second option in molecular fingerprinting. In this method, [small subgraphs of a molecule are mapped to a massive one hot vector encoding for possible subgraph structures. ][]This vector is hashed down to a reasonable length, say 1024 bits. This approach assigns near unique identifiers to molecules and allows measurements of distance (Hamming for instance). In this way, it avoids the possible degeneracy of molecular characteristics. However, it is no longer continuous and the hashing makes it impossible to explore feature importance etc. The most recent advances in molecular embeddings have focused on learned embeddings. Particularly those using [graph convolutional neural networks.][] This methods provide a fully differentiable network which learns in concert with a classifier to generate embeddings from arbitrary molecular graphs. These embeddings are both unique and continuous, and can provide a rough feature importance interpretation. Thus I expect that they will provide substantial improvements in agent performance when compared with previous techniques.

### GHD Comments	

> I have data for this, in the over/undersampling data set. I need to dig through it again, but I think that the take away is slightly better performance from the concated versions. Which makes sense. 

### C) Deal with Imbalanced Data

A crucial feature of HTS is the extreme imbalance between active and inactive compounds. Therefore, concurrently with the exploration above, I need to find techniques for ameliorating the effects of this. At the moment, predicting Inactive will yield ~ 90% or greater accuracy. I have a few ideas about this, some of which are data based, some are related to the agents themselves.

From [imbalanced-learn][]  

+ Upsample (repeatedly sample from Actives): This doesn't really provide new information, just pushes agent to consider classes more equitably. Some argue that this is just an inefficient way to do class weighting.
+ Downsample (selectively remove some inactives): people usually don't like this because it is seen as 'throwing away data' )
+ [SMOTE][Smote Explained] : In this approach, you synthetically up sample the minority class. In essence, interpolating between Actives to generate more actives. I'm a bit nervous about this. For molecular fingerprints, you can't really interpolate between hash bits in any logical manner. Other embeddings should make more sense, but given how small changes in a molecule can yield changes in activity I'm a bit leery of labeling synthesize data as 'Active'

Loss techniques:

+ [Focal Loss][Focal Loss] This loss function down weights the contribution of easily classified examples : 										$$FL = -\sum_{i=1}^{C=2}(1 - s_{i})^{\gamma }t_{i} log (s_{i})$$

  This only works with probabilistic outputs from the classifier so DNNs are the primary use case. For DNNs one needs to modify the bias of the output layer so that you don't get divergent loss on the first round. The details are quoted from the paper below, but the initialization is partially dependent on an estimate of the class weight of the minority class. In our case, we don't know the ratios of active to inactive for each library/target pairing so may need to use averages etc

  >  For the final convlayer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π), where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π. We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, this initialization prevents the large number of background anchors from generating a large, destabilizing loss value in the first iteration of training.

There are some examples of [XGBoost with Focal Loss](<https://github.com/jhwjhw0123/Xgboost-With-Imbalance-And-Focal-Loss>). 

+ Cohen's Kappa: This is another loss/metric that reweights based on class imbalance. 
+ F1 Loss

###  GHD comments

> The biggest example here is in the multiple_metrics_df_over_under.pkl file which I ran the 80/10 split with both over and under sampling on all three RDKit derived embedding. There are also some files that show my attempts with DNNs and [focal loss](https://www.comet.ml/gdreiman1/dnn-baseline/b8272505413d474c8cb1217c62bd1b82?experiment-tab=Other), but I basically found that there were no noticeable benefits from using the focal loss, I think because there were enough 'hard' negatives that it focused on them as well as any positives that were missed. Also because positives were so few, there were gradients that would explode on occasion because they'd be so wrong that loss ~ $\infin $. Also was never comfortable with smote on chemicals because it doesn't make sense at all on MFPs and it 

## 2) Build Baseline Iterative Approach

After benchmarking the proposed classifiers and embeddings in step 1, the most successful models will be moved over to an iterative approach. This approach still needs to be fleshed out but the basic plan is as follows:

​	 1) Initial Screen

​		This step provides the agent with initial information about the library and the target. Initially I intend to take a 10% sample at random and train on it. There may be some merit to doing a k-nn clustering to try to sample an informative sub-space. [There are other initialization approaches that I may also consider][https://www.aidanf.net/publications/atem-03finn.pdf].

​	2) Iteration

​		In this step, the partially trained agent makes predictions on the remaining unscreened portion of the library. These predictions are the basis for selecting further library members to screen. Initially, I intend to select the compounds that are predicted to be active with the highest confidence levels. However, there are other potential heuristics for selecting the next batch. These include 'doping' by adding a selection of predicted inactives (probably selecting the inactives with the lowest confidence). Additionally, there is [evidence][] that weak reinforcement (selectively screening low confidence samples) increases yields. In any case, I intend to base some of the characteristics of the iterative algorithm  on the technical characteristics of HTS (i.e. choosing batch sizes to match with multiples of plate size etc).

### GHD comments

| Description                                                  | Comet Link                                                   | Graph Location        |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------- | ---- |
| Select top 4% of ranked, plus 1% (random), 10 iterations, 3 replicates, stored hist for NNs. |                                                              |                       |      |
| Select top 4% of ranked, plus 1% (diversity if iter <5 else random), 10 iterations, 3 replicates, stored hist for NNs. | [first 8 AID](https://www.comet.ml/gdreiman1/iter-baseline/4860d6df1427456284e52e3b9d919ef5) , [893](https://www.comet.ml/gdreiman1/iter-baseline/38b6a6e11c80433dbbacb925727aaa0a) | /Thesis/Photos/Iter_9 |      |
| Greedy only policy                                           | [1](https://www.comet.ml/gdreiman1/iter-baseline/65f5b1125fb44669b52574a0e14fa51d) [2](https://www.comet.ml/gdreiman1/iter-baseline/6a902abbbd3f4be089efbd5da7fbf97c) [3](https://www.comet.ml/gdreiman1/iter-baseline/6ef6209f0bc643bbbc0288c9c18cc00d) | /Thesis/Photos/Iter_8 |      |
| 100 epochs, 5% iter size, Diverse Selection, No Weak In-actives after predicted actives falls below 80% of batch size | [link](https://www.comet.ml/gdreiman1/iter-baseline/b6632c0543b548aa97896610fac507ba) | /Thesis/Photos/Iter_7 |      |
| 100 Epochs, 10% iter size, Random Selection, [top 7% and bottom 1% of active and top 1% inactive] or [all active, 50%weak 50%random] includes GCNN | [link](https://www.comet.ml/gdreiman1/iter-baseline/78fa184784914681800204941cbfd2b0) | /Thesis/Photos/Iter_6 |      |
| 10% iter size, random selection, 10 epochs at GCNN           | [link](https://www.comet.ml/gdreiman1/iter-baseline/55e7e5fa093d43d9a96375482980b885) | /Thesis/Photos/Iter_5 |      |
| 5% iter, diverse, no GCNN, initial selection strat           | [link](https://www.comet.ml/gdreiman1/iter-baseline/2cac28b1474b483da186dba01ed71be7) | /Thesis/Photos/Iter_4 |      |
| 5% iter, random, no GCNN, initial selection strat            | [link](https://www.comet.ml/gdreiman1/iter-baseline/42f4146aef974009bc1e4f431691d579) | /Thesis/Photos/Iter_3 |      |
| 10% iter, random, no GCNN, initial selection strat           | [link](https://www.comet.ml/gdreiman1/iter-baseline/f078ae1056844f44a21debb691ed8118) | /Thesis/Photos/Iter_2 |      |
|                                                              |                                                              |                       |      |
|                                                              |                                                              |                       |      |

> 10 total iterative experiments, I've basically played with:
>
> Active Selection $\in \{\text{above threshold, rank and take fixed %}\}$
>
> Iteration size $\in \{\text{5%, 10%}\}$
>
> Exploration type $\in \{\text{diverse relative to already scanned,random, select near decision boundary }\}$
>
> Explore vs Exploit ratio $\in \{\text{greedy,1:4,weird custom combo}\}$
>
> They all take a really long time to run, driven mostly by the graphs I think. It seems like ranking really removed any benefits that GCNNs were showing, (this is reinforced by plotting (% active found) vs (% lib scanned)). Basically they were just much more likely to predict above the 0.5 threshold that I set.
>
> The things I'd like to still investigate:
>
> + 10% vs 15% starting point
> + How long do I really need to train the GCNNs 100 epochs is just kinda an arbitrary point.
> + Using different embedding for diversity
> + Making GCNNs more efficient!  
>   + Either pretrain or retain 

### 2B) Evaluate on inhouse data/use in an initial trial run

The final aspect of the Baseline is attempting to apply the results to a selection of data from the Drug Discovery Institute itself.  If it looks promising in these tests, I'd like to get it to a functional state where it could be used in future HTS campaigns.

### GHD comments

> Not done yet!!
>
> Not sure if we want to 



## 3) Expand to more exotic approaches (the fun stuff)

I think that this situation is well suited to reinforcement learning. In particular, I really would like to try the [Active One-Shot Learning](<https://arxiv.org/abs/1702.06559>) approach. I really like this approach because it removes the human made heuristics from designing the iterative algorithm. Instead, an agent learns when to request a label for itself. Thus we can pre-train the agent on the masses of available data, and in the process it should learn the optimal strategy for deciding which samples to select. Then we can apply this pre-trained model to a novel library/target combination and allow it to finetune its sample selection protocol to fit the individual data.

I also am interested in more broadly exploring [One Shot Learning](<https://www.tandfonline.com/doi/full/10.1080/17460441.2019.1593368?af=R>)and intend to try some of the methods mentioned in the review. There is already an older paper[ Low Data Drug Discovery with One-Shot Learning](<https://pubs.acs.org/doi/full/10.1021/acscentsci.6b00367>) which seems to implement very similar one-shot approaches to learning activity in the same library with different targets.  Unfortunately, it seems that they were partially unable to exceed the benchmarks they set with Random Forests. However, they succeed in using Siamese and AttnLSTMs to exceed the RF benchmark by a wide margin.

### GHD comments

> Haven't done much in this regard. I think that I intend to use the translation embedding on a bit of a moonshot with AOSL





## Thoughts at this point

### Major Concerns:

1. Never hyper parameter tuned anything.
   1. The counter argument is that it wasn't really clear what drove best performance, so hard to quantify. 
   2. But almost everyone who has a paper that I've read does actually tune them, usually for acc or for AUC
2. Lots of descisions were just kinda seat of the pants
   1. GCNN is modified from a github repo reproducing DEEPCHEM graphconvmodel in pytorch
3. Active learning strats were not throughly mapped:
   1. I just kinda tried the uncertainty vs random vs diverse methods at whim
   2. I can argue that I needed to primarily focus on exploitation because we wanted to validate things that we predict active

### Left to do

1. For iterative approaches
   1. 10% vs 15% starting point
   2. How long do I really need to train the GCNNs 100 epochs is just kinda an arbitrary point.
   3. Using different embedding for diversity
   4. Making GCNNs more efficient!  
      - Either pretrain or retain 
   5. Test on own data?
2. For prior approaches 
   1. Either collate old records better or just run once more the early 80-10 experiments to get some more metrics. Currently not Prec/Rec curves
3. Fancy approaches
   1.  AOSL with translated embeddings
   2. Try GIN arch?



[Molecular Char Degeneracy]: https://www.sciencedirect.com/science/article/pii/S1359644617304695#bib0090

[There are other initialization approaches that I may also consider]: https://www.aidanf.net/publications/atem-03finn.pdf

[graph convolutional neural networks.]: https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints
[small subgraphs of a molecule are mapped to a massive one hot vector encoding for possible subgraph structures. ]: https://pubs.acs.org/doi/pdf/10.1021/ci100050t
[evidence]: https://www.ncbi.nlm.nih.gov/pubmed/25915687
[imbalanced-learn]: https://imbalanced-learn.readthedocs.io/en/stable/
[Focal Loss]: <https://arxiv.org/abs/1708.02002>
[Smote Explained]: <http://rikunert.com/SMOTE_explained>