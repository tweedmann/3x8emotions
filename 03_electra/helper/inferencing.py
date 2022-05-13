import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import helper.training as tr


class Inferencer:
    def __init__(self):
        self.emotions = [
            "anger",
            "fear",
            "disgust",
            "sadness",
            "joy",
            "enthusiasm",
            "pride",
            "hope",
        ]
        self.MODEL_NAME = "german-nlp-group/electra-base-german-uncased"
        self.DIR_TRAINED_MODEL = "./models/final"
        self.SEED = 7
        set_seed(self.SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"{self.DIR_TRAINED_MODEL}/{self.MODEL_NAME}", num_labels=8
        ).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def predict(self, x):
        val = []
        for record in x:
            # tokenize document
            inputs = self.tokenizer(
                record, truncation=True, padding=True, return_tensors="pt"
            )
            inputs = inputs.to(device=self.device)
            # inference
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.sigmoid()
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            prediction = prediction.detach().cpu().numpy()
            val.append(prediction[0])
        return np.array(val)

    def predict_dataframe(self, x):
        predictions = self.predict(x)
        list_for_df = []
        for i in range(len(x)):
            row = [*[x[i]], *predictions[i]]
            list_for_df.append(row)
        columns = ["text"] + self.emotions
        return pd.DataFrame(list_for_df, columns=columns)
