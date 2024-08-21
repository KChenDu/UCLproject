from pathlib import Path
from json import loads
from human_eval.data import write_jsonl


if __name__ == '__main__':
    file_names = ('MBPP(Python)_nucleus92_batch1.jsonl', 'MBPP(Python)_nucleus92_batch2.jsonl', 'MBPP(Python)_nucleus92_batch3.jsonl', 'MBPP(Python)_nucleus92_batch4.jsonl')

    root = Path(__file__).parent / 'data'
    merged = []

    with open(root / 'dsc13B-base_mbpp_codon_feedback_greedy.jsonl', 'r') as file:
        for line in file:
            datum = loads(line)
            datum['attempt'] = 0
            merged.append(datum)

    offset = 1

    for file_name in file_names:
        with open(root / file_name, 'r') as file:
            for line in file:
                datum = loads(line)
                datum['sample'] += offset
                merged.append(datum)
        offset += 20

    write_jsonl('dsc13B-base_mbpp_codon_feedback_merged.jsonl', merged)
