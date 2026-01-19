import argparse
import asyncio
from pathlib import Path
from collections import defaultdict
from openai import AsyncOpenAI
from datasets import load_from_disk
from prompt import *

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline LLMs (Async)")
    p.add_argument("--k", type=int, default=10, help="Number of answers per query")
    p.add_argument("--k_per_query", type=int, default=1, help="Answers per request")
    p.add_argument("--output_filedir", type=str, default="./results/baselines/")
    p.add_argument("--filename", type=str, default="list")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini")
    p.add_argument("--max_concurrency", type=int, default=20)
    p.add_argument("--temperature", type=float, default=None)
    return p.parse_args()


async def main_async(args):
    client = AsyncOpenAI()

    dataset = load_from_disk("~/CLAN/DivergeRAG/data/clan_diverge_dataset")["train"]
    dataset = dataset.select(range(100))
    queries = dataset["prompt"]

    Path(args.output_filedir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_filedir) / f"{args.llm_model}_{args.filename}_multi_turn.txt"

    sem = asyncio.Semaphore(args.max_concurrency)

    async def query_worker(qid: int, query: str):
        """
        One query → K serial generations (conversation-style)
        """
        answers_all = []

        # ---- initialize conversation ----
        messages = [
            {
                "role": "user",
                "content": llm_prompt_multi_turn.format(
                    QUESTION=query,
                ),
            }
        ]

        async with sem:
            for t in range(args.k):
                kwargs = dict(model=args.llm_model, input=messages)
                if args.temperature is not None:
                    kwargs["temperature"] = args.temperature

                try:
                    resp = await client.responses.create(**kwargs)
                    ans = resp.output_text.strip()
                    answers_all.append(ans)

                    # ---- append conversation context ----
                    messages.append(
                        {
                            "role": "assistant",
                            "content": ans,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "For the same question, give me another answer."
                            ),
                        }
                    )

                except Exception as e:
                    print(f"[WARN] qid={qid}, step={t} failed: {e}")

        return qid, answers_all

    # ---- query-level parallelism ----
    tasks = [
        query_worker(qid, query)
        for qid, query in enumerate(queries)
    ]
    results = await asyncio.gather(*tasks)

    # ---- write output ----
    counter = defaultdict(int)
    lines = []

    for qid, answers in results:
        for ans in answers:
            counter[qid] += 1
            ans = ans.replace("\n", " ").strip()
            lines.append(f"{counter[qid]}|{qid+1}: {ans}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()