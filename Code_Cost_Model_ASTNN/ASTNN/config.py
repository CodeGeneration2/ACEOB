VOCAB_SIZE = None    # Size of the Word2Vec vocabulary, None means no limit.
MIN_COUNT = 3        # Ignore all words with a total frequency lower than this value.
EMBEDDING_SIZE = 128  # Embedding size of the word vectors.
RATIO = "3:1:1"      # The ratio for splitting the dataset into training, validation, and test sets.
HIDDEN_DIM = 100     # ST-hidden dimension of the ST-Tree encoder.
ENCODE_DIM = 128     # ST-hidden dimension of the BiGRU encoder.
LABELS = 104         # Number of output classes.
EPOCHS = 10
BATCH_SIZE = 64      # Batch size of 64
USE_GPU = True

TASK = "Time Prediction"    # "Time Prediction" or "Source Code Classification"
