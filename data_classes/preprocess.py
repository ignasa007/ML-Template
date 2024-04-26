# any functions needed to preprocess data
# this is different from collate function in that collate processes a batch of samples at a time
# preprocessing is stuff like cleaning up text -- 
#       it is the same in each run (unlike, say, random augmentation) and
#       does not require context from other samples (unlike, say, batch norm)