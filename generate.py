import json
import argparse
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="google/gemma-2-2b")
parser.add_argument("--table_form", type=str, default="table_md")
parser.add_argument("--lora", type=bool, default=False)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()


prompt = """주어진 질문에 대한 정확한 답을 테이블에서 찾아주세요. 아래 테이블에 나와 있는 항목만을 기반으로 답변해야 하며, 답변은 간결하게 작성해주세요. 종결어(예: '입니다', '입니다.' 등)는 생략해주세요.

### 테이블:
{table}

### 질문: 
{question}?

### 답변:
"""


eval_ds = datasets.load_dataset("json", data_files='./data/ds_test.json')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
)
if args.lora:
    model = PeftModel.from_pretrained(model, f"lora_adapter_{args.table_form}")
    model = model.merge_and_unload()
model.eval()

def format_prompt(ds):
    text = prompt.format(table=ds[args.table_form], question=ds['question'])
    return {'prompt': text}

eval_ds = eval_ds.map(format_prompt)
def data(ds):
    for i in tqdm(range(len(ds))):
        yield ds[i]['prompt']

pipe = pipeline(model=model, task='text-generation', tokenizer=tokenizer, max_new_tokens=32, return_full_text=False)
generations = []
for out in pipe(data(eval_ds['train'])):
    generations.append(out[0]['generated_text'])

with open('./data/ds_test.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

results = []
for gen, item in zip(generations, dataset):
    new_item = {
        args.table_form: item[args.table_form],
        'question': item['question'],
        'answer': item['answer'],
        'generation': gen
    }
    results.append(new_item)
with open(args.save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
