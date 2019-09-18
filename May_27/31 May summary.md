# 31 May summary

+ I have the Comet ML up and running, seems that is a great way to store things.

+ Have 6 data sets processed, but not stored. I guess I should get 10 total? Not sure.
  + Need to think about: is it useful to have some from same library to try to do meta/transfer learning? Maybe, maybe not
+ Classifiers:
  + Next week want to run thru all classifiers against the 6 so far
  + lightgbm seems good, paradoxically has better recall on actives with fewer learners
    + Also can think about using kappa or MCC based objectives for this (there's a kaggle kernel of MCC)
  + neural nets, seem actually pretty promising. using elus with drop out seemed good. I need to get the $\pi$ bias prior going for focal loss versions. The loss seems to diverge w big batch sizes. Also GPUs!!
    + Can think about using either sagemaker or theanos to do some hyper paramters searching. I think inital bad performance was bc bad tuning and maybe too small network
+ Found the imbalanced data library
  + I think that this should be helpful, need to try some of the sythenthetic approaches?
+ Want to think more about what the implications of class labels are for binclass. 
  + Discuss w fredrik how to balance recall vs precision
+ Probably should read the big backlog of papers
  + Focus on those that directly handle hts prediction, active learning can wait
+ Need better compute quickly. Maybe time to reactive GCP? But then I need to learn the damn thing
  + Also look at the UCL compute servers. 

+ Try training some regressors, what do they do?
+ 