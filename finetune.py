from typing import List
import os
import fire
import torch
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from dotenv import load_dotenv
from datasets import Dataset
import torch.nn.functional as F

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from sklearn.metrics import average_precision_score

from transformers import LlamaForCausalLM, LlamaTokenizer

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf", 
    output_dir: str = "output/",

    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    val_set_size: int = 2000,
    
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
):

    device_map = "auto"


    # Step 1: Load the model and tokenizer

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True, # Add this for using int8
        torch_dtype=torch.float16,
        device_map=device_map,
        token=hf_token
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model, token=hf_token)
    tokenizer.pad_token_id = 0

    #Add this for training LoRA

    config = LoraConfig(
          r=lora_r,
          lora_alpha=lora_alpha,
          target_modules=lora_target_modules,
          lora_dropout=lora_dropout,
          bias="none",
          task_type="CAUSAL_LM",
      )
    model = get_peft_model(model, config)

    model = prepare_model_for_int8_training(model) # Add this for using int8


    # Step 2: Load the data

    train_data = pd.read_csv("data/train.csv")
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1123)
    
    # Step 3: Tokenize the data
    def one_hot_encode(label):
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        one_hot = np.zeros(5)
        one_hot[mapping[label]] = 1
        return one_hot
    
    def tokenize(row):
        input_string = f"Question: {row['prompt']} sentenceA: {row['A']} sentenceB: {row['B']} sentenceC: {row['C']} sentenceD: {row['D']} sentenceE: {row['E']}"
        input_ids = tokenizer.encode(input_string)
        labels = one_hot_encode(row['answer'])
        return pd.Series({"input_ids": input_ids, "labels": labels})
    
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)

    # Apply tokenization
    train_data = train_data.apply(tokenize, axis=1)
    val_data = val_data.apply(tokenize, axis=1)

    train_data = Dataset.from_pandas(train_data)
    val_data= Dataset.from_pandas(val_data)
    
    # Step 4: Initiate the trainer

    # 평균 정밀도 계산
    def mean_average_precision(y_true, y_score):
        aps = []
        for i in range(y_true.shape[0]):
            ap = average_precision_score(y_true[i], y_score[i])
            aps.append(ap)
        return sum(aps) / len(aps)

    # 메트릭 계산
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = F.softmax(logits, dim=-1)  # 로짓을 확률로 변환
        map_score = mean_average_precision(labels.cpu().numpy(), probs.cpu().detach().numpy())
        return {"map": map_score}
    
    
    # 임의의 입력 텐서 생성 (입력 차원에 따라 변경 가능)
    
    # # 모델의 출력 확인
    # with torch.no_grad():
    #     dummy_output = model(dummy_input)

    # print("Dummy output shape:", dummy_output.shape)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        # compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()


    # Step 5: save the model
    model.save_pretrained(output_dir)




if __name__ == "__main__":
    fire.Fire(train)