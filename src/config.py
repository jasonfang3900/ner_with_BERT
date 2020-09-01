import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE= 64
VALID_BATCH_SIZE = 4
EPOCHS = 10
BASE_MODEL_PATH = "../pretrained_encoders/bert-base-uncased"
OUTPUT_MODEL_PATH = "../saved_models"
TRAINING_FILE = "../data/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
