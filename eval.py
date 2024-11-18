import argparse
import json
import numpy as np

from qa_metrics.em import em_match
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
# nltk.download('punkt_tab')

def sim_score(targets, generations, sim_model):
    targets_embedding = sim_model.encode(targets)
    generations_embedding = sim_model.encode(generations)
    
    cos_sims = []
    for i in range(targets_embedding.shape[0]):
        similarities = cosine_similarity(targets_embedding[i].reshape(1,-1), generations_embedding[i].reshape(1,-1))
        cos_sims.append(similarities[0][0])
    cos_sim = float(np.mean(cos_sims, axis = 0))
    return round(cos_sim, 4)

def bleu_score(targets, generations):
    bleu_scores = []
    for pred, truth in zip(generations, targets):
        pred_tokens = nltk.word_tokenize(pred)
        truth_tokens = nltk.word_tokenize(truth)

        smooth = SmoothingFunction()
        score = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=smooth.method1)
        bleu_scores.append(score)
    
    bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    return round(bleu, 4)

def f1_score(targets, generations):
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    num_sentences = len(targets)

    for pred, truth in zip(generations, targets):
        words_truth = set(truth.split())
        words_pred = set(pred.split())

        tp = len(words_truth.intersection(words_pred))
        fp = len(words_truth - words_pred)
        fn = len(words_pred - words_truth)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_f1 += f1_score
        total_precision += precision
        total_recall += recall

    avg_f1 = total_f1 / num_sentences if num_sentences > 0 else 0
    avg_precision = total_precision / num_sentences if num_sentences > 0 else 0
    avg_recall = total_recall / num_sentences if num_sentences > 0 else 0

    return {
        'avg_f1': round(avg_f1, 4), 
        'avg_precision': round(avg_precision, 4), 
        'avg_recall': round(avg_recall, 4)
    }

def em_match_score(targets, generations):
    res = []
    for pred, truth in zip(generations, targets):
        res.append(em_match([truth], pred))
    em = sum(res) / len(res)
    return round(em, 4)


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", type=str)
parser.add_argument("-s", "--save_name", type=str)
parser.add_argument("-m", "--metrics", type=str, nargs='+', help="f1, exact_match, similarity, bleu")
args = parser.parse_args()

with open(args.file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)
targets = [d['answer'] for d in data]
generations = [d['generation'] for d in data]

results = {}
for metric in args.metrics:
    if metric == 'similarity':
        sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        score = sim_score(targets, generations, sim_model)
    elif metric == 'bleu':
        score = bleu_score(targets, generations)
    elif metric == 'f1':
        score = f1_score(targets, generations)
    elif metric == 'exact_match':
        score = em_match_score(targets, generations)

    results[metric] = score

with open(f"./data/eval/{args.save_name}.json", 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
