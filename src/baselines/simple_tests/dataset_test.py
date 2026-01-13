from datasets import load_from_disk

dataset = load_from_disk("~/CLAN/DivergeRAG/data/clan_diverge_dataset")["train"]
print (dataset)