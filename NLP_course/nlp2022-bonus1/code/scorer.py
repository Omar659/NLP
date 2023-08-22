from argparse import ArgumentParser
from pprint import pprint
from typing import Dict


def parse_file(path) -> Dict[str, str]:
    id2ans: Dict[str, str] = dict()
    with open(path) as lines:
        for line in lines:
            id, answer = line.strip().split("\t")
            id2ans[id] = answer
    return id2ans


def evaluate(answers, golds):
    correct: int = 0
    tot: int = 0
    for id in golds.keys():
        ans: str = answers[id]
        label: str = golds[id]
        if ans == label:
            correct += 1
        tot += 1
    accuracy: float = (correct / tot) * 100
    err_rate: float = 100 - accuracy
    return dict(err_rate=f"{err_rate:.2f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--gold_file", required=True)

    args = parser.parse_args()

    prediction_file = args.prediction_file
    gold_file = args.gold_file

    id2answer = parse_file(prediction_file)
    id2gold = parse_file(gold_file)
    results = evaluate(id2answer, id2gold)

    pprint(results)
