from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import DebertaV2ForMultipleChoice, DebertaV2Tokenizer
import numpy as np
import torch.nn.functional as F

# Data loading
train_data = pd.read_csv('data/train.csv')
# all_data = pd.read_csv('data/train.csv')

extra_6000_data = pd.read_csv('data/6000_train_examples.csv')
extra_data = pd.read_csv('data/extra_train_set.csv')

# Combine all the datasets
all_data = pd.concat([train_data, extra_6000_data, extra_data], ignore_index=True)

# Data Preprocessing for all_data
all_prompts = all_data['prompt'].tolist()
all_choices = all_data[['A', 'B', 'C', 'D', 'E']].values.tolist()
all_answers = [ord(a) - ord('A') for a in all_data['answer'].tolist()]

all_data['A'].fillna('Unknown', inplace=True)
all_data['B'].fillna('Unknown', inplace=True)
all_data['C'].fillna('Unknown', inplace=True)
all_data['D'].fillna('Unknown', inplace=True)
all_data['E'].fillna('Unknown', inplace=True)

all_choices = all_data[['A', 'B', 'C', 'D', 'E']].applymap(str).values.tolist()

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = '/kaggle/input/devertav3-large/kaggle/input/deberta_v3_large'
model_path = 'microsoft/deberta-v3-large'
model = DebertaV2ForMultipleChoice.from_pretrained(model_path, return_dict=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

all_input_ids_list = []
all_attention_mask_list = []

for i in range(len(all_prompts)):
    prompt = all_prompts[i]
    choice_list = all_choices[i]
    prompt_choice_pairs = [(prompt, choice) for choice in choice_list]
    # print(type(prompt), [type(choice) for choice in choice_list])

    inputs = tokenizer.batch_encode_plus(
        prompt_choice_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    all_input_ids_list.append(inputs["input_ids"])
    all_attention_mask_list.append(inputs["attention_mask"])

max_len = max(tensor.shape[1] for tensor in all_input_ids_list)
padded_input_tensors = [F.pad(input=tensor, pad=(0, max_len - tensor.shape[1])) for tensor in all_input_ids_list]
padded_attention_tensors = [F.pad(input=tensor, pad=(0, max_len - tensor.shape[1])) for tensor in all_attention_mask_list]

all_input_ids = torch.stack(padded_input_tensors)
all_attention_mask = torch.stack(padded_attention_tensors)
all_labels = torch.tensor(all_answers, dtype=torch.long)

# Split the data into training and validation sets
train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(all_input_ids, all_attention_mask, all_labels, test_size=0.1, random_state=42)

train_dataloader = DataLoader(TensorDataset(train_input_ids, train_attention_mask, train_labels), batch_size=2, shuffle=True)
val_dataloader = DataLoader(TensorDataset(val_input_ids, val_attention_mask, val_labels), batch_size=2, shuffle=False)


best_map3 = 0.0  # Initialize best MAP@3

from sklearn.preprocessing import LabelBinarizer



# Fine-tuning with validation
def fine_tune_and_validate(model, train_dataloader, val_dataloader, epochs=10):
    global best_map3
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        
        # Training
        model.train()
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                outputs = model(input_ids, attention_mask=attention_mask)[0]
                scores = torch.softmax(outputs, dim=1)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        

        
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # One-hot encode the labels
        lb = LabelBinarizer()
        lb.fit(all_labels)
        all_labels_onehot = lb.transform(all_labels)
        # print("One-hot encoded labels shape:", all_labels_onehot.shape)  # Should be (20, 5) in your case

        map3 = label_ranking_average_precision_score(all_labels_onehot, all_scores)
        print(f"Validation MAP@3 for epoch {epoch+1}: {map3}")
        
        if map3 > best_map3:
            print("New best model. Saving...")
            best_map3 = map3
            model.save_pretrained("extra_deberta_v3_large")
            tokenizer.save_pretrained("extra_deberta_v3_large")
    return model

# Fine-tuning the model
model = fine_tune_and_validate(model, train_dataloader, val_dataloader, epochs=10)