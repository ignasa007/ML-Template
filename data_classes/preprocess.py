'''
Any methods/classes needed to preprocess the data.
This is different from collate function in that collate processes a batch of samples at a time.
Preprocessing is stuff like cleaning up text:
    - It is the same in each run (unlike, say, random augmentation) and
    - It does not require context from other samples (unlike, say, batch norm)
'''