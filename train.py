import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer

BASE_MODEL = 'bert-base-cased'
LEARNING_RATE = 1e-1
MAX_LENGTH = 512  # change this
BATCH_SIZE = 40
EPOCHS = 20

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL,
                                                           num_labels=1)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25
                    ]) / len(single_squared_errors)

    return {'mse': mse, 'mae': mae, 'r2': r2, 'accuracy': accuracy}


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='../models/camembert-fine-tuned-regression-2',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    metric_for_best_model='accuracy',
    load_best_model_at_end=True,
    weight_decay=0.01,
)


def preprocess_function(data):
    label = float(data['price'])
    label = np.log(label)
    input_text = data['features']
    output = tokenizer(input_text,
                       truncation=True,
                       padding='max_length',
                       max_length=256)
    output['label'] = label
    return output


if __name__ == '__main__':
    y_train_raw = pd.read_csv('data/tabular/y_train_OXxrJt1.csv')
    full_df = pd.read_csv('data/image_captions/df.csv')
    full_df['price'] = y_train_raw['price']

    train, test = train_test_split(full_df, test_size=0.2)

    train = train[:3000]
    test = test[:300]
    ds = {'train': train, 'validation': test}

    for split in ds:
        ds[split] = ds[split].apply(lambda row: preprocess_function(row),
                                    axis=1)
        ds[split].reset_index(inplace=True, drop=True)

    class RegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs[0][:, 0]
            loss = torch.nn.functional.mse_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        compute_metrics=compute_metrics_for_regression,
    )

    trainer.train()
