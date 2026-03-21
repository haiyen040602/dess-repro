import json
from pathlib import Path

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def count_entities(samples):
    return sum(len(s.get("entities", [])) for s in samples)

def main():
    base_dir = Path("data")
    datasets = ["14res", "14lap", "15res", "16res"]
    splits = ["train", "dev", "test"]

    print("DATASET STATISTICS (Sentences & Entities)")
    print("-" * 78)
    print(f"{'Dataset':<8} {'Split':<8} {'Sentences':<12} {'Entities':<12}")
    print("-" * 78)

    for dataset in datasets:
        dataset_total_sent = 0
        dataset_total_ent = 0
        for split in splits:
            file_path = base_dir / dataset / f"{split}_dep_triple_polarity_result.json"
            if not file_path.exists():
                print(f"{dataset:<8} {split:<8} {'MISSING':<12} {'MISSING':<12}")
                continue
            samples = load_json(file_path)
            sent_count = len(samples)
            ent_count = count_entities(samples)
            dataset_total_sent += sent_count
            dataset_total_ent += ent_count
            print(f"{dataset:<8} {split:<8} {sent_count:<12} {ent_count:<12}")

        print(f"{dataset:<8} {'TOTAL':<8} {dataset_total_sent:<12} {dataset_total_ent:<12}")
        print("-" * 78)

if __name__ == "__main__":
    main()
