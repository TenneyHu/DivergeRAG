from divrag import DivRAG
from datasets import DatasetDict


dataset = DatasetDict.load_from_disk("./data/clan_diverge_dataset")["train"]
nums_answers = 10

with open("diverge_results.txt", "w") as f:
    for qid, data in enumerate(dataset):
        query = data["prompt"]
        div = DivRAG(query=query, qid=qid, embed_model="text-embedding-3-small", llm_model="gpt-5-mini", max_generation_num=nums_answers)
        results = div.run()
        ans = []
        for i, res in enumerate(results):
            res = res.replace("\n", " ").replace("\t", " ").strip()
            f.write(f"Qid {qid}\tNum{i+1}\t{res}\n")

