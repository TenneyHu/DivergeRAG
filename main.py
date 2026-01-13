from divrag import DivRAG
from datasets import DatasetDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

file_lock = threading.Lock()


def run_one_query(qid, data, nums_answers, outfile):
    query = data["prompt"]

    div = DivRAG(
        query=query,
        qid=qid,
        embed_model="text-embedding-3-small",
        llm_model="gpt-5-mini",
        max_generation_num=nums_answers,
    )

    results = div.run()

    cleaned_results = [
        res.replace("\n", " ").replace("\t", " ").strip()
        for res in results
    ]

    with file_lock:
        for i, res in enumerate(cleaned_results):
            outfile.write(f"{i+1}|{qid+1}:\t{res}\n")
        outfile.flush()   

    return qid


async def run_all_queries_async(dataset, nums_answers, outfile, max_workers=4):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    tasks = []
    for qid, data in enumerate(dataset):
        task = loop.run_in_executor(
            executor,
            run_one_query,
            qid,
            data,
            nums_answers,
            outfile,
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


dataset = DatasetDict.load_from_disk("./data/clan_diverge_dataset")["train"]
dataset = dataset.select(range(100))
nums_answers = 10
MAX_WORKERS = 30

with open("./results/diverge_results1.txt", "w", encoding="utf-8") as f:
    asyncio.run(
        run_all_queries_async(
            dataset=dataset,
            nums_answers=nums_answers,
            outfile=f,
            max_workers=MAX_WORKERS,
        )
    )
