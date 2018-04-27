"""
Global Constant
"""

import string

EMBEDDING_SIZE = 300
WORD_INDEXER_SIZE = 307915
SEQUENCE_LENGTH =2000
DROPOUT_PROB = 0.5

NUM_FILTER = (2,2)
FILTER_WIDTH = (3,2)

HIDDEN_DIM = 50

# Spacebar
SPACEBAR = " "

# Escape Character
ESCAPE_WORD_DELIMITER = "\t"
ESCAPE_TAG_DELIMITER = "\v"

# Tag
PAD_TAG_INDEX = 0

# TAG_LIST = ['POS','NEU','NEG']
TAG_LIST = ['POS','NEG']

TAG_START_INDEX = 0


NUM_TAGS = len(TAG_LIST)

DEFAULT_MODEL_PATH = "models/cnn_multi_2tag_e3.h5"
DEFAULT_WORD_INDEX_PATH = 'word_index.pickle'

# Random Seed
SEED = 1395096092


