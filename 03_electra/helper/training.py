import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import Trainer, TrainingArguments

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
    
    
def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()

def weighted_f1_loss(y_pred, y_true, weight=2):
    y_pred = torch.from_numpy(y_pred)
    y_pred = y_pred.sigmoid()
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0

    loss = 0
    f1_scores = []
    for i in range(len(y_true[0])):
        f1 = f1_score(y_true[:,i],y_pred.int().numpy()[:,i])
        f1_scores.append(f'{f1:9.4f}')
        loss += weight*(1 -f1)
    #print(loss, f1_scores)
    return loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy_thresh_value = accuracy_thresh(predictions, labels)
    weighted_f1_loss_value = weighted_f1_loss(predictions, labels)
    return {'accuracy_thresh': accuracy_thresh_value, 'f1_loss':weighted_f1_loss_value}    
    
    
def compute_fine_metrics2(eval_pred,emotions):
    metrics_result = {
        "f1": [],
        "precision": [],
        "recall": [],
        "f1_micro": [],
        "f1_macro": [],
        "f1_weighted": [],
    }
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = torch.tensor(predictions)

    preds_full = torch.sigmoid(predictions).cpu().detach().numpy().tolist()

    preds_full = np.array(preds_full) >= 0.5
    labels = np.array(labels) >= 0.5

    for i, label in enumerate(emotions):
        column_preds = preds_full[:, i]
        column_labels = labels[:, i]
        prf1 = metrics.precision_recall_fscore_support(
            column_labels, column_preds, average="binary"
        )
        metrics_result["f1"].append(prf1[2])
        metrics_result["precision"].append(prf1[0])
        metrics_result["recall"].append(prf1[1])
        metrics_result["f1_micro"].append(
            metrics.f1_score(column_labels, column_preds, average="micro")
        )
        metrics_result["f1_macro"].append(
            metrics.f1_score(column_labels, column_preds, average="macro")
        )
        metrics_result["f1_weighted"].append(
            metrics.f1_score(column_labels, column_preds, average="weighted")
        )

    return metrics_result    

def compute_metrics_single(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    f1_score_micro = f1_score(labels, preds, average='micro')
    f1_score_macro = f1_score(labels, preds, average='macro')
    f1_score_weighted = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_micro': f1_score_micro,
        'f1_macro': f1_score_macro,
        'f1_weighted': f1_score_weighted,
    }