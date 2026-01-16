import hashlib
import os
import json
import numpy as np
from openai import OpenAI as OpenAIClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.anthropic import Anthropic
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from search import google_search
from logger import setup_debug_logger
from prompt import *
import hashlib
import threading
import argparse
from datasets import DatasetDict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading

file_lock = threading.Lock()

QUERY_CACHE_DIR = "/dtu/p1/tianyhu/cache/queries"
os.makedirs(QUERY_CACHE_DIR, exist_ok=True)

def index_check(index, query):
    retriever = index.as_retriever(similarity_top_k=5)
    base_nodes = retriever.retrieve(query)
    if len(base_nodes) > 0:
        return True
    else:
        return False

def setup_llm(provider: str, model: str):
    if provider == "openai":
        Settings.llm = OpenAIClient(model=model)
    elif provider == "claude":
        Settings.llm = Anthropic(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
def query_cache_dir(query: str) -> str:
    h = hashlib.md5(query.encode("utf-8")).hexdigest()
    return os.path.join(QUERY_CACHE_DIR, f"q_{h}")

     

class RAG:
    def __init__(
        self, 
        query: str,
        embed_model: str,
        llm_model: str,
        max_generation_num: int = 5,
        retrieval_k:int=5,
        retrieval_chunk_size:int=512,
        chunk_overlap:int=50,
        option = "rag"
    ):

        self.client = OpenAIClient()
        self.query = query
        self.query_num = 5
        self.retrieval_k= retrieval_k
        self.max_generation_num = max_generation_num
        Settings.chunk_size = retrieval_chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        self.use_mmr = False
        self.use_shuffle = False
        self.use_expand = False
        if option == "mmr" or option == "all":
            self.use_mmr = True
        if option == "shuffle" or option == "all":
            self.use_shuffle = True
        if option == "expand" or option == "all":
            self.use_expand = True

        self.llm_model = llm_model
        self.logger = setup_debug_logger(
            log_dir="./logs",
            log_name="baselinerag.log",
        )
        if llm_model.startswith("claude"):
            Settings.llm = Anthropic(model=llm_model)
        else:
            Settings.llm = OpenAI(model=llm_model)
        self.steps = 0

    def run(self):
        self.logger.info(f"Running RAG for query: {self.query}")
        self.logger.info(f"MMR: {self.use_mmr}, Shuffle: {self.use_shuffle}, Expand: {self.use_expand}")

        if not self.use_expand:
            index = self._search(self.query)
            return self._rag(index)

        queries = self._expand(self.query)
        self.logger.info(f"Expanded queries: {queries}")
        
        all_nodes = []
        seen_ids = set()

        for q in queries:
            index_q = self._search(q)
            if index_q is None:
                continue
                
            try:
                nodes = list(index_q.docstore.docs.values())

            except AttributeError:
                nodes = index_q if isinstance(index_q, list) else []

            for n in nodes:
                nid = getattr(n, "id_", getattr(n, "doc_id", None))
                if nid and nid not in seen_ids:
                    all_nodes.append(n)
                    seen_ids.add(nid)



        final_index = VectorStoreIndex(all_nodes, embed_model=Settings.embed_model)
        response = self._rag(final_index)
        return response
    
    def _expand(self, query: str):
        expansion_prompt = PromptTemplate(expansion_prompt_template)
        expansion_prompt = expansion_prompt.format(QUERY=query, k=self.query_num - 1)
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": expansion_prompt}
            ]
        )
        raw = resp.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            expanded_queries = data["queries"]
            if not isinstance(expanded_queries, list):
                raise ValueError("queries is not a list")
        except Exception as e:
            raise ValueError(f"Failed to parse expanded queries:\n{raw}") from e
        expanded_queries.append(query)
        return expanded_queries

    def _search(self, query: str):
        qdir = query_cache_dir(query)
        if os.path.exists(qdir):
            storage_context = StorageContext.from_defaults(
                persist_dir=qdir
            )
            index = load_index_from_storage(storage_context)
            if index_check(index, query):
                return index

        search_results = google_search(
            query,
            num_results=self.retrieval_k,
            min_chars=128,
            verbose=True,
            logger=self.logger
        )

        documents = []
        for result in search_results:
            if isinstance(result, dict):
                text = result.get("text", "").strip()
                if not text:
                    continue

                doc = Document(
                    text=text,
                    metadata={"url": result.get("url")},
                )
                documents.append(doc)

            elif isinstance(result, str) and result.strip():
                doc = Document(text=result)
                documents.append(doc)

        index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
        index.storage_context.persist(persist_dir=qdir)

        return index

    def _rag(self, index: VectorStoreIndex):
        if self.use_mmr:
            retriever = index.as_retriever(
                similarity_top_k=self.retrieval_k,
                vector_store_query_mode="mmr",
            )
        else:
            retriever = index.as_retriever(
                similarity_top_k=self.retrieval_k,
            )

        base_nodes = retriever.retrieve(self.query)

        response_synthesizer = get_response_synthesizer(
            text_qa_template=PromptTemplate(rag_prompt),
            streaming=False
        )

        def generate_one(trial_id: int):
            local_nodes = list(base_nodes)
            if self.use_shuffle:
                r = random.Random(trial_id)
                r.shuffle(local_nodes)
            response = response_synthesizer.synthesize(
                query=self.query,
                nodes=local_nodes
            )
            return str(response)

        responses = []

        GEN_WORKERS = min(5, self.max_generation_num)

        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as executor:
            futures = [
                executor.submit(generate_one, t)
                for t in range(self.max_generation_num)
            ]

            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    self.logger.error(f"Generation error: {e}")

        return responses


def run_one_query(qid, data, nums_answers, option, outfile):
    query = data["prompt"]

    div = RAG(
        query=query,
        embed_model="text-embedding-3-small",
        llm_model="gpt-5-mini",
        option=option,
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


async def run_all_queries_async(dataset, nums_answers, option, outfile, max_workers=4):
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
            option,
            outfile,
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results/baselines/", help="Output directory for results")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model to use for generation")
    parser.add_argument("--option", type=str, default="rag", help="Option name for output file")
    
    args = parser.parse_args()
                        
    dataset = DatasetDict.load_from_disk("./data/clan_diverge_dataset")["train"]
    dataset = dataset.select(range(100))
    nums_answers = 10
    MAX_WORKERS = 5

    with open(os.path.join(args.output_dir, f"{args.model}_{args.option}.txt"), "w", encoding="utf-8") as f:
        asyncio.run(
            run_all_queries_async(
                dataset=dataset,
                nums_answers=nums_answers,
                option=args.option,
                outfile=f,
                max_workers=MAX_WORKERS,
            )
        )
if __name__ == "__main__":
    main()

        
