import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from prompt import *
import json
import asyncio
from openai import AsyncOpenAI

def lexical_diversity(text_dict, max_n=3):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    per_input_avg_ratios = []

    for _, texts in text_dict.items():
        uniq_ratios = []

        for n in range(1, max_n + 1):
            all_ngrams = []
            for text in texts:
                tokens = text.strip().split()
                all_ngrams.extend(get_ngrams(tokens, n))
            total = len(all_ngrams)
            unique = len(set(all_ngrams))
            ratio = unique / total if total > 0 else 0
            uniq_ratios.append(ratio)

        avg_ratio = sum(uniq_ratios) / len(uniq_ratios) if uniq_ratios else 0
        per_input_avg_ratios.append(avg_ratio)
    print("Lexical diversity:", sum(per_input_avg_ratios) / len(per_input_avg_ratios) if per_input_avg_ratios else 0.0)
    return sum(per_input_avg_ratios) / len(per_input_avg_ratios) if per_input_avg_ratios else 0.0

def semantic_diversity_openai(qid_to_texts):
    diversities = []
    client = OpenAI()
    for texts in qid_to_texts.values():
        if len(texts) < 2:
            continue
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        # shape: (num_texts, dim)
        embeddings = torch.tensor(
            [d.embedding for d in response.data],
            dtype=torch.float32,
        )

        norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm, norm.t())
        idx = torch.triu_indices(len(texts), len(texts), offset=1)
        upper_triangular = similarity_matrix[idx[0], idx[1]]
        diversity = 1 - upper_triangular.mean().item()
        diversities.append(diversity)
    print ("Semantic diversity:", np.mean(diversities) if diversities else 0.0)
    return float(np.mean(diversities)) if diversities else 0.0

VERDICT_TO_ID = {
    "Excellent": 5,
    "Good": 4,
    "Fair": 3,
    "Poor": 2,
    "Irrelevant": 1,
}

async def quality_score_async(args, queries, max_concurrency=20):
    """
    Async evaluation of answer quality for open-ended questions.

    Args:
        args: namespace or dict containing
              - model (model name)
              - quality_prompt (prompt template string)
        queries: dict
            { qid: (question: str, answers: List[str]) }
        max_concurrency: int, max number of concurrent API calls

    Returns:
        float: average quality score
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def eval_one(qid, question, answer):
        async with semaphore:
            prompt = args.quality_prompt.format(
                QUESTION=question,
                ANSWER=answer,
            )

            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)

                verdict = parsed.get("verdict", "Unsound")
                reason = parsed.get("reason", "Invalid JSON output")

            except Exception:
                verdict = "Unsound"
                reason = "Failed to parse JSON output"

            verdict_id = VERDICT_TO_ID.get(verdict, 0)
            print(f"QID: {qid}, Verdict: {verdict} ({verdict_id}), Reason: {reason}")

            return verdict_id

    tasks = []
    for qid, (question, answers) in queries.items():
        for ans in answers:
            tasks.append(eval_one(qid, question, ans))

    verdict_ids = await asyncio.gather(*tasks)

    final_score = float(np.mean(verdict_ids)) if verdict_ids else 0.0
    print("Average quality score:", final_score)

    return final_score

def parse_results_file(filepath):
    queries = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Qid 0\tNum1\tXXX
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            # 解析 qid
            qid = int(parts[0].split()[1])

            answer = parts[2].strip()
            if answer == "Empty Response":
                print("Empty Response found, skipping.")
                continue
            if qid not in queries:
                queries[qid] = []

            queries[qid].append(answer)

    return queries

def parse_file(filepath):
    qid_to_texts = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: num|qid: answer
            try:
                _, rest = line.split('|', 1)
                qid_part, answer = rest.split(':', 1)
                qid = int(qid_part)
            except ValueError:
                continue

            answer = answer.strip()
            if answer == "Empty Response":
                print("Empty Response found, skipping.")
                continue
            if qid not in qid_to_texts:
                qid_to_texts[qid] = []

            qid_to_texts[qid].append(answer)

    return qid_to_texts

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def query_relevance_async(
    args,
    queries,
    max_concurrency=20,
):

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_and_score(qid, original_query, text):
        async with semaphore:

            prompt = args.query_generation_prompt.format(ANSWER=text)

            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)
                new_query = parsed["question"]


                emb_resp = await client.embeddings.create(
                    model=args.embed_model,
                    input=[original_query, new_query],
                )

                orig_emb = emb_resp.data[0].embedding
                new_emb = emb_resp.data[1].embedding

                sim = cosine_sim(orig_emb, new_emb)

            except Exception as e:
                new_query = ""
                sim = 0.0

            print(
                f"QID {qid} | sim={sim:.3f} | original: {original_query} | generated: {new_query}"
            )

            return {
                "qid": qid,
                "original_query": original_query,
                "generated_query": new_query,
                "similarity": sim,
            }

    tasks = []

    for qid, (orig_query, texts) in queries.items():
        for text in texts:
            tasks.append(
                generate_and_score(qid, orig_query, text)
            )


    results = await asyncio.gather(*tasks)

    return np.mean([res["similarity"] for res in results])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="gpt-5", help="LLM model name")
    parser.add_argument("--quality_prompt", type=str, default=quality_prompt, help="Quality evaluation prompt template")
    parser.add_argument("--query_generation_prompt", type=str, default=question_generation_prompt, help="Query generation prompt template")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="Embedding model name")
    args = parser.parse_args()

    dataset = load_from_disk("~/CLAN/DivergeRAG/data/clan_diverge_dataset")["train"]
    raw_prompts = dataset["prompt"]                    


    qid_to_texts = parse_file("./results/baselines/gpt-5-mini_vsampling.txt")
    #qid_to_texts = parse_file("./results/diverge_results.txt")
    queries = {
        qid: (raw_prompts[qid - 1], texts)
        for qid, texts in qid_to_texts.items()
        if 1 <= qid <= 21
    }
 



    
    final_score = asyncio.run(
        quality_score_async(args, queries, max_concurrency=20)
    )
    print("Final quality score:", final_score / 5.0)
    
    
    lex_div = lexical_diversity(qid_to_texts, max_n=3)
    semantic_div = semantic_diversity_openai(qid_to_texts)