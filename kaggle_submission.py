from transformers import DebertaV2ForMultipleChoice, DebertaV2Tokenizer, DebertaV2Config
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# 데이터 로드
# data = pd.read_csv('data/train.csv')
data = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = '/kaggle/input/devertav3-large/kaggle/input/deberta_v3_large'
model = DebertaV2ForMultipleChoice.from_pretrained(model_path, return_dict=True)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model.to(device)

# 데이터 전처리
prompts = data['prompt'].tolist()
choices = data[['A', 'B', 'C', 'D', 'E']].values.tolist()
# answers = [ord(a) - ord('A') for a in data['answer'].tolist()]

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

# DataLoader 생성을 위한 수정
dataset = TensorDataset(input_ids, attention_mask)  # answers 부분이 제거됨
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)  # shuffle을 False로 설정

# 예측 결과를 저장할 DataFrame 생성
submission_data = []

# 예측 함수 정의
def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            input_ids, attention_mask = batch

            # 텐서를 GPU로 이동
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)[0]
            scores = torch.softmax(outputs, dim=1)
            
            # GPU에서 CPU로 데이터를 이동하고 NumPy 배열로 변환
            scores = scores.cpu().numpy()

            for i in range(scores.shape[0]):  # 추가: 각 배치에 있는 모든 샘플에 대한 예측을 수행
                pred_scores = scores[i]
                sorted_indices = pred_scores.argsort()[::-1][:3]
                sorted_choices = " ".join([chr(ord('A') + j) for j in sorted_indices])
                actual_id = idx * dataloader.batch_size + i  # 실제 데이터 샘플의 id 계산
                submission_data.append([actual_id, sorted_choices])

# 예측 수행
predict(model, dataloader)

# submission.csv 파일로 저장
submission_df = pd.DataFrame(submission_data, columns=['id', 'prediction'])
# submission_df.to_csv('submission.csv', index=False)

# # 'id' 열을 DataFrame의 인덱스로 설정
# submission_df.set_index('id', inplace=True)

# CSV 파일로 저장 (이번에는 index=True)
submission_df.to_csv('submission.csv', index=False)