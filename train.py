import argparse
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--table_form", type=str)
parser.add_argument("--save_path", type=str, default="./model")
args = parser.parse_args()

prompt = """주어진 질문에 대한 정확한 답을 테이블에서 찾아주세요. 아래 테이블에 나와 있는 항목만을 기반으로 답변해야 하며, 답변은 간결하게 작성해주세요. 종결어(예: '입니다', '입니다.' 등)는 생략해주세요.

### 테이블:
{table}

### 질문: 
{question}?

### 답변:
"""


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.padding_side = 'right'

ds = datasets.load_dataset("json", data_files='./data/ds_train.json')
def format_prompt(ds):
    text = prompt.format(table=ds[args.table_form], question=ds['question'])
    return {'prompt': text}
ds = ds.map(format_prompt)
ds = ds.shuffle()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    quantization_config=bnb_config,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=args.save_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    group_by_length=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)
model.config.use_cache = False
trainer.train()

model.config.use_cache = True
trainer.model.save_pretrained(f"lora_adapter_{args.table_form}")

# merged = PeftModel.from_pretrained(model, "lora_adapter", device_map='auto')
# merged = merged.merge_and_unload()
# merged.push_to_hub("gemma-2-2b-tune")

# messages = [{"role": "user", "content": "Hello doctor, I have bad and painfull acne on face and body. How can I get rid of it?"}]
# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

# # Optimized generation with tuned sampling strategies
# outputs = model.generate(
#     **inputs,
#     max_length=350,  # Increase max length for complex answers
#     num_return_sequences=1,
#     top_k=50,
#     top_p=0.85,  # Narrow top-p for more deterministic output
#     temperature=0.3,  # Slightly higher temperature for balance between creativity and accuracy
#     no_repeat_ngram_size=3,
# )

# # Decode and clean up the output
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# response = text.split("assistant")[1].strip()

# print(response)