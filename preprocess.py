import re
import random
import json
import hashlib

import html_to_json
from langchain_core.documents import Document
from langchain_community.document_transformers import MarkdownifyTransformer, Html2TextTransformer


def save_json(data, file_name):
    with open(f"./data/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def prep_html_tags(table):
    cleaned = re.sub(r'<br\s*/?>', '\n', table)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    return cleaned

def push_index(dataset): # 테이블 인덱스 생성
    table_hash_map = {}
    for item in dataset:
        table_content = item["table"]
        table_hash = hashlib.md5(table_content.encode('utf-8')).hexdigest()

        if table_hash not in table_hash_map:
            table_hash_map[table_hash] = len(table_hash_map)
        item["table_index"] = table_hash_map[table_hash]

def table_transform(done_idx, docs, dataset, transformer, form_type='md'): # Html table to Markdown / Text
    transformed_docs = transformer.transform_documents(docs)
    result = {}
    for table_index, doc in zip(done_idx, transformed_docs):
        result[table_index] = doc.page_content.replace("\\", "")
    
    with open(f'./data/check_{form_type}.text', 'w') as f:
        f.write('\n$$$\n'.join(list(result.values())))

    for item in dataset:
        item[f'table_{form_type}'] = result[item['table_index']]

def table_to_json(done_idx, docs, dataset): # Html table to Json
    result = {}

    for table_index, doc in zip(done_idx, docs):
        json_table = html_to_json.convert_tables(doc.page_content)[0]
        json_doc = {key: [] for key in json_table[0]}
        for table in json_table[1:]:
            for key, val in zip(json_table[0], table):
                json_doc[key].append(val)
        result[table_index] = json_doc

    for item in dataset:
        item["table_json"] = result[item['table_index']]

def split_data(dataset):
    num_data = {}
    for item in dataset:
        if item['table_index'] not in num_data:
            num_data[item['table_index']] = 0
        num_data[item['table_index']] += 1
    
    eval_inds = [table_index for table_index, num in num_data.items() if num == 4]
    eval_inds = random.sample(eval_inds, 50)
    valid_inds, test_inds = eval_inds[:5], eval_inds[5:]
    train, test, valid = [], [], []
    for item in dataset:
        if item['table_index'] in valid_inds:
            valid.append(item)
        elif item['table_index'] in test_inds:
            test.append(item)
        else:
            train.append(item)
    
    random.shuffle(train)
    return train, test, valid


with open('./data/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
push_index(data)

done_idx, docs = [], []
for item in data:
    if item['table_index'] in done_idx:
        continue
    done_idx.append(item['table_index'])
    docs.append(Document(item['table']))

print("###### table to json")
table_to_json(done_idx, docs, data)
table_transform(done_idx, docs, data, MarkdownifyTransformer(), 'md')
print("###### table to md")
table_transform(done_idx, docs, data, Html2TextTransformer(), 'text')
print("###### table to text")
save_json(data, 'dataset_transform')

print("###### splitting data")
train, test, valid = split_data(data)
save_json(train, 'ds_train')
save_json(test, 'ds_test')
save_json(valid, 'ds_valid')
