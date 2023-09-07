from transformers import DebertaV2ForMultipleChoice, DebertaV2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# 데이터 로드
# data = pd.read_csv('data/train.csv')
data = pd.read_csv('data/test.csv')

base_model = "microsoft/deberta-v3-large"
model = DebertaV2ForMultipleChoice.from_pretrained(base_model)
tokenizer = DebertaV2Tokenizer.from_pretrained(base_model)

# 데이터 전처리
prompts = data['prompt'].tolist()
choices = data[['A', 'B', 'C', 'D', 'E']].values.tolist()
answers = [ord(a) - ord('A') for a in data['answer'].tolist()]

# Update the tokenization step
input_ids_list = []
attention_mask_list = []
for i in range(len(prompts)):
    prompt = prompts[i]
    choice_list = choices[i]
    # Creating a list of 5 pairs: each prompt-choice pair as a tuple
    prompt_choice_pairs = [(prompt, choice) for choice in choice_list]
    
    inputs = tokenizer.batch_encode_plus(
        prompt_choice_pairs,
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids_list.append(inputs["input_ids"])
    attention_mask_list.append(inputs["attention_mask"])

# Step 1: Determine max length
max_length = 0
for ids in input_ids_list:
    max_length = max(max_length, ids.size(1))

# Step 2: Pad each tensor to max length
padded_input_ids_list = []
padded_attention_mask_list = []
for ids, mask in zip(input_ids_list, attention_mask_list):
    padding_length = max_length - ids.size(1)
    
    # Pad input_ids
    padded_ids = torch.cat([
        ids, 
        torch.zeros((ids.size(0), padding_length), dtype=ids.dtype)
    ], dim=1)
    padded_input_ids_list.append(padded_ids)
    
    # Pad attention_mask
    padded_mask = torch.cat([
        mask,
        torch.zeros((mask.size(0), padding_length), dtype=mask.dtype)
    ], dim=1)
    padded_attention_mask_list.append(padded_mask)

# Step 3: Stack the tensors
input_ids = torch.stack(padded_input_ids_list)
attention_mask = torch.stack(padded_attention_mask_list)

# DataLoader 생성
dataset = TensorDataset(input_ids, attention_mask, torch.tensor(answers))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 평가 함수 정의
def evaluate_MAP(model, dataloader):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)[0]
            scores = torch.softmax(outputs, dim=1)
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
    
    # MAP 계산
    total_precision = 0
    num_questions = len(all_labels)
    
    for i in range(num_questions):
        scores = np.array(all_scores[i])
        correct_label = all_labels[i]
        
        # 예측 점수에 따라 정렬
        sorted_indices = np.argsort(scores)[::-1]
        
        # Precision@1 계산
        if sorted_indices[0] == correct_label:
            total_precision += 1
            
    map_score = total_precision / num_questions
    return map_score

# 예측 결과를 저장할 DataFrame 생성
submission_data = []

# 평가 함수 정의
def evaluate_and_predict(model, dataloader):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)[0]
            scores = torch.softmax(outputs, dim=1)
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            
            # 예측 결과를 저장합니다
            pred_scores = scores[0]
            sorted_indices = pred_scores.argsort(descending=True)
            sorted_choices = " ".join([chr(ord('A') + i) for i in sorted_indices.tolist()])
            submission_data.append([idx, sorted_choices])

# submission.csv 파일로 저장
submission_df = pd.DataFrame(submission_data, columns=['id', 'prediction'])
submission_df.to_csv('submission.csv', index=False)

# # 기본 능력 테스트
# map_score = evaluate_MAP(model, dataloader)
# print(f"Mean Average Precision: {map_score}")