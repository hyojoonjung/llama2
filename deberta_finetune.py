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
import bitsandbytes as bnb

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

from transformers import Trainer, DataCollatorWithPadding, PretrainedConfig, \
                        DebertaForTokenClassification, DebertaTokenizer, DebertaForCausalLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class CustomDebertaForClassification(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, 5)
        
        # 새로운 코드: Custom 모델의 설정을 정의합니다.
        self.config = base_model.config

        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        return logits

    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

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
    probs = F.softmax(logits, dim=-1)
    predicted_indices = torch.argmax(probs, dim=1).cpu().numpy()
    one_hot_labels = np.eye(5)[predicted_indices]  # Convert to one-hot format
    map_score = mean_average_precision(labels.cpu().numpy(), one_hot_labels)
    return {"map": map_score}

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = bnb.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, torch.argmax(labels, dim=1))
        return (loss, outputs) if return_outputs else loss

# def custom_loss(output, labels):
#     # output: [batch_size, sequence_length, vocab_size]
#     # labels: [batch_size, 5]  (5 is the number of choices A, B, C, D, E)
#     output_for_choices = output[:, -1, :5]  # Assuming the choices are represented by the first 5 tokens in the vocabulary and you are interested in the last token of the sequence
#     loss = F.cross_entropy(output_for_choices, torch.argmax(labels, dim=1))
#     return loss

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss = custom_loss(logits, labels)
#         return (loss, outputs) if return_outputs else loss


def train(
    # model/data params
    base_model: str = "microsoft/deberta-base", 
    output_dir: str = "output/",

    micro_batch_size: int = 1,
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

    model = DebertaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        # (여기에 DeBERTa에 적합한 다른 옵션을 추가할 수 있습니다)
    )
    model = CustomDebertaForClassification(model)  # 변경된 부분

    tokenizer = DebertaTokenizer.from_pretrained(base_model)  # 변경된 부분
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
    
    def compute_loss(model, batch, return_outputs=False):
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(logits, torch.argmax(batch['labels'], dim=1))
        return (loss, logits) if return_outputs else loss
    
    # 임의의 입력 텐서 생성 (입력 차원에 따라 변경 가능)
    
    # # 모델의 출력 확인
    # with torch.no_grad():
    #     dummy_output = model(dummy_input)

    # print("Dummy output shape:", dummy_output.shape)
    
    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     compute_metrics=compute_metrics,
    #     compute_loss=compute_loss,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=100,
    #         num_train_epochs=num_epochs,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=10,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=200,
    #         save_steps=200,
    #         output_dir=output_dir,
    #         save_total_limit=3
    #     ),
    #     data_collator=DataCollatorWithPadding(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    trainer = CustomTrainer(
    model=model,
    args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            # optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3
        ),
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

    trainer.train()


    # Step 5: save the model
    model.save_pretrained(output_dir)




if __name__ == "__main__":
    fire.Fire(train)