import argparse
import asyncio
import os
from pathlib import Path
import json
import aiohttp
from openai import AsyncOpenAI
from datasets import load_from_disk
from prompt import *
from collections import defaultdict

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline LLMs (Async)")
    p.add_argument("--k", type=int, default=1, help="Number of trials per prompt")
    p.add_argument("--k_per_query", type=int, default=10, help="Number of trials per query")
    p.add_argument("--input", type=str, default="./data/novelty-bench.txt", help="Path to input file")
    p.add_argument("--output_filedir", type=str, default="./results/baselines/", help="Output directory")
    p.add_argument("--filename", type=str, default="vsampling", help="Output file name")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="OpenAI model name")
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"], help="LLM provider to use")
    p.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent requests")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature for LLM")
    return p.parse_args()


async def main_async(args):
    client = None

    if args.provider == "openai":
        client = AsyncOpenAI()
    elif args.provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for provider=claude")
    else:
        raise RuntimeError(f"Unknown provider: {args.provider}")



    dataset = load_from_disk("~/CLAN/DivergeRAG/data/clan_diverge_dataset")["train"]
    dataset = dataset.select(range(100))
    queries = dataset["prompt"]
    
    Path(args.output_filedir).mkdir(parents=True, exist_ok=True)
    if args.temperature is not None:
        output_file = Path(args.output_filedir) / f"{args.llm_model}_{args.temperature}.txt"
    else:
        output_file = Path(args.output_filedir) / f"{args.llm_model}_{args.filename}.txt"

    sem = asyncio.Semaphore(args.max_concurrency)

    async def one_request(idx, trial, query, k_per_query=args.k_per_query):
        async with sem:
            prompt = vs_prompt.format(QUESTION=query, K=k_per_query)

            answers = []  

            if args.provider == "openai":
                kwargs = dict(model=args.llm_model, input=prompt)
                if args.temperature is not None:
                    kwargs["temperature"] = args.temperature

                try:
                    resp = await client.responses.create(**kwargs)
                    raw = resp.output_text.strip()

                    data = json.loads(raw)
                    parsed = data.get("answers", [])

                    answers = []
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "text" in item:
                                answers.append(item["text"])

                    else:
                        answers = []

                except Exception as e:
                    print(f"[WARN] Failed to parse response for qid={idx}, trial={trial}: {e}")
                    answers = []

            try:
                k = int(k_per_query)
                if len(answers) > k:
                    answers = answers[:k]
            except Exception:
                pass  

        return idx, trial, answers



    tasks = [
        one_request(idx, trial, query, args.k_per_query)
        for idx, query in enumerate(queries)
        for trial in range(args.k)
    ]
    results = await asyncio.gather(*tasks)


    counter = defaultdict(int)
    lines = []

    for qid, _, answers in results:
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