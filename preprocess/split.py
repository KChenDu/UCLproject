from argparse import ArgumentParser
from pathlib import Path
from os import cpu_count
from datasets import load_dataset
from json import loads
from context import time_limit, no_stdout
from human_eval.data import write_jsonl


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    path = Path(args.path)
    assert path.is_file()

    task_id2test = {}
    dataset = load_dataset("mbpp", split="train", num_proc=cpu_count())

    for data in dataset:
        task_id = data['task_id']
        data.pop('task_id')
        task_id2test[task_id] = data

    compilable_dataset = []
    not_compilable_dataset = []

    with open(path, 'r') as file:
        for line in file:
            data = loads(line)
            if data['compilable']:
                if data['attempt'] != 0:
                    not_compilable_dataset.append(data)
                data.pop('compilable')
                task_id = data['task_id']
                code = data['generation'] + "\n\n" + task_id2test[task_id]['test_setup_code'] + "\n\n"
                test_list = task_id2test[task_id]['test_list']
                try:
                    for test in test_list:
                        with time_limit(3.), no_stdout():
                            exec(code + test + '\n', {})
                        data['pass'] = True
                except BaseException as e:
                    data['pass'] = False
                compilable_dataset.append(data)
            else:
                not_compilable_dataset.append(data)

    write_jsonl(str(path.with_suffix('')) + '_compilable.jsonl', compilable_dataset)
    write_jsonl(str(path.with_suffix('')) + '_not_compilable.jsonl', not_compilable_dataset)
