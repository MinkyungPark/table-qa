import json
import numpy as np
import matplotlib.pyplot as plt

names = ['html', 'markdown', 'text', 'json', 'fine-tune-markdown']
files = ['table_pt.json', 'table_md_pt.json', 'table_text_pt.json', 'table_json_pt.json', 'table_md_sft.json']

em_scores = []
for file in files:
    with open(f"./data/eval/{file}", 'r') as f:
        data = json.load(f)
    em_scores.append(data['exact_match'])

colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
plt.bar(names, em_scores, color=colors)
plt.xlabel('Models')
plt.ylabel('Exact Match Score')
# plt.title('')
plt.savefig('./data/plot.png') 