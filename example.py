from divrag import DivRAG
from datasets import DatasetDict
import torch
import numpy as np
from openai import OpenAI

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
        print ("Similarity matrix upper triangular values:", upper_triangular)
        diversity = 1 - upper_triangular.mean().item()
        diversities.append(diversity)

    return float(np.mean(diversities)) if diversities else 0.0

def load_demo():
    dataset = DatasetDict.load_from_disk("./data/clan_diverge_dataset")["train"]
    dataset = dataset[1]
    return dataset

def run_demo():
    query = load_demo()["prompt"]
    print ("Prompt:", query)

    nums_answers = 5

    # Initialize DivRAG

    div = DivRAG(
        query=query,
        qid=0,
        embed_model="text-embedding-3-small",
        llm_model="gpt-5-mini",
        max_generation_num=nums_answers,
        retrieval_chunk_size=512,
        debug=True
    )

    results = div.run()
    ans = []
    print("DIVERGE Results:")
    for i, res in enumerate(results):
        res = res.replace("\n", " ").strip()
        print(f"Answer {i+1}: {res}")
        ans.append(res)
    return ans

ans = run_demo()

qid_to_texts = {
    "demo_query": ans
}

diversity_score = semantic_diversity_openai(qid_to_texts)
print(f"\nSemantic Diversity Score (OpenAI embeddings): {diversity_score:.4f}")