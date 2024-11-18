
### package used
```
torch transformers langchain tqdm openpyxl openai pandas datasets langchain-community ragatouille qa-metrics bitsandbytes accelerate markdownify html2text peft trl
```

### Preprocess
```
python3 preprocess.py
```

### Train
```
python3 train.py --table_form table_md
```
- `--table_form`
    - table : original html table
    - table_md : Markdown
    - table_text : Text
    - table_json : Json

### Generate
```
python3 generate.py --save_path ./data/generate/table_md_sft.json --table_form table_md --lora=True
```
- `--lora`
    - False : gemma2-2b
    - True : LoRA fine-tuned model


### Evaluation
```
python3 eval.py --file_name ./data/generate/table_md_sft.json --save_name table_md_sft --metrics f1 exact_match similarity bleu
```
- `--file_name`
    - generations from generate.py
- `--save_name`
    - results with eval score
- `--metrics` : eval score metrics
    - exact_match
    - f1
    - bleu
    - similarity ('sentence-transformers/all-MiniLM-L6-v2’)

