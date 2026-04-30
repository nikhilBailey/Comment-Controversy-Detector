#This computes Cohen's Kappa inter-annotator agreement between two annotators on the first 100 comments of the cleaned MrBeast dataset.
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


#Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "intermediate_data")

EVAN_FILE   = os.path.join(DATA_DIR, "beast_cleaned_evan_first100.txt")
NIKHIL_FILE = os.path.join(DATA_DIR, "beast_cleaned_nikhil_first100.txt")


def load_labels(filepath: str) -> list:
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            #rsplit on the last comma so commas inside the comment text don't break parsing
            try:
                _, label_str = line.rsplit(",", 1)
                labels.append(int(label_str))
            except ValueError:
                print(f"  Warning: could not parse label on line {line_num} of {os.path.basename(filepath)}")
    return labels

def main() -> None:
    evan_labels = load_labels(EVAN_FILE)
    nikhil_labels = load_labels(NIKHIL_FILE)

    if len(evan_labels) != len(nikhil_labels):
        print("Not equal amount of labels, exiting now!")
        exit()

    kappa = cohen_kappa_score(evan_labels, nikhil_labels)

    n = len(evan_labels)
    classes = sorted(set(evan_labels) | set(nikhil_labels))

    #Confusion matrix (rows = Evan, cols = Nikhil)
    cm = confusion_matrix(evan_labels, nikhil_labels, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Evan={c}" for c in classes],
        columns=[f"Nikhil={c}" for c in classes],
    )

    print("Cohen's Kappa Inter-Annotator Agreement")
    print(f"Comments compared:    {n}")
    print(f"Cohen's Kappa:        {kappa:.4f}")
    print("\nConfusion matrix (rows = Evan, cols = Nikhil):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
