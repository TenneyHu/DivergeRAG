#liweijiang/infinite-chats-taxonomy

from datasets import DatasetDict, load_dataset
from tqdm import tqdm

def user_prompt(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def extract_categories(categories):
    return [c["category"] for c in categories if "category" in c]


def save():
    dataset = load_dataset("liweijiang/infinite-chats-taxonomy")
    ds = dataset["train"]  
    ds = ds.map(lambda ex: {"prompt": user_prompt(ex["messages"]), "category_list": extract_categories(ex["categories"])}, remove_columns=ds.column_names)

    predefined_categories = {
    "Problem Solving",
    "Decision Support",
    "Concept Explanations",
    "Skill Development",
    "Recommendations",
    "Opinion-Based Questions",
    "Value-Laden Questions",
    "Controversial Questions",
    "Ideation and Brainstorming",
    "Personal Advice"
    }
    #count the number of each category
    subset_ds = DatasetDict({"train": ds.select([])})

    for i in tqdm(range(len(ds))):
        positive_count = 0
        count = 0
        for category in ds[i]["category_list"]:
            if category in predefined_categories:
                positive_count += 1
            count += 1
        if positive_count == count:
            subset_ds["train"] = subset_ds["train"].add_item(ds[i])
            if len(subset_ds["train"]) >= 200:
                break


    print(subset_ds)
    #dump the dataset
    subset_ds.save_to_disk("./data/clan_diverge_dataset")

def load():
    dataset = DatasetDict.load_from_disk("./data/clan_diverge_dataset")["train"]
    for i in range(10):
        print (dataset[i])

if __name__ == "__main__":
    load()