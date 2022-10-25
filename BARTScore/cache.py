from transformers import (AutoModel, AutoTokenizer, RobertaTokenizer)

model_type = "roberta-large" # default
tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir="./models/roberta", use_fast=False, do_lower_case=True)
model = AutoModel.from_pretrained(model_type, cache_dir="./models/roberta")