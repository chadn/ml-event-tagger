"""Configuration and constants."""

# Tag taxonomy (19 tags)
TAGS = [
    # Music genres & performers (10 tags)
    "music", "house", "techno", "breaks", "jazz", "rock", "punk", "hiphop", "dj", "band",
    # Activities (4 tags)
    "dance", "yoga", "art", "food",
    # Access & venue (5 tags)
    "outdoor", "indoor", "public", "private", "free",
    # Other characteristics (2 tags)
    "weekly", "community"
]

# Model hyperparameters
MAX_VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 100
DENSE_UNITS = 32
BATCH_SIZE = 16
EPOCHS = 30

# Training parameters
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Paths
MODEL_DIR = "models"
DATA_DIR = "data"

