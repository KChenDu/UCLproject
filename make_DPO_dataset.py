import signal

from contextlib import contextmanager
from signal import setitimer, ITIMER_REAL, signal, SIGALRM
from datasets import load_dataset
from os import cpu_count
from json import loads
from random import choice
from json import dump


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    setitimer(ITIMER_REAL, seconds)
    signal(SIGALRM, signal_handler)
    try:
        yield
    finally:
        setitimer(ITIMER_REAL, 0)


if __name__ == '__main__':
    task_id2test = {}
    dataset = load_dataset("mbpp", split="train", num_proc=cpu_count())

    for data in dataset:
        task_id = data['task_id']
        data.pop('task_id')
        task_id2test[task_id] = data

    task_id2reference = {}

    for task_id, test in task_id2test.items():
        task_id2reference[task_id] = test['code']

    task_id2samples = {}

    with open('MBPP(Python)_nucleus92_3attempts.jsonl', 'r') as f:
        for line in f:
            data = loads(line)
            data.pop('attempt')
            sample = data['sample']
            data.pop('sample')
            task_id = data['task_id']
            data.pop('task_id')
            if task_id not in task_id2samples:
                task_id2samples[task_id] = [[data]]
            elif sample >= len(task_id2samples[task_id]):
                task_id2samples[task_id].append([data])
            else:
                task_id2samples[task_id][sample].append(data)

    task_id2positives = {}
    task_id2pairs = {}

    for task_id, samples in task_id2samples.items():
        test_setup_code = task_id2test[task_id]['test_setup_code']
        for sample in samples:
            if sample[-1]['compilable']:
                generation = sample[-1]['generation']
                code = generation
                if len(test_setup_code) > 0:
                    code += "\n\n" + test_setup_code
                test_list = task_id2test[task_id]['test_list']
                try:
                    for test in test_list:
                        with time_limit(3.):
                            exec(code + "\n\n" + test + '\n', {})
                    if task_id in task_id2positives:
                        task_id2positives[task_id].append(generation)
                    else:
                        task_id2positives[task_id] = [generation]
                    for attempt in sample[:-1]:
                        if task_id in task_id2pairs:
                            task_id2pairs[task_id].append((generation, attempt['generation']))
                        else:
                            task_id2pairs[task_id] = [(generation, attempt['generation'])]
                except BaseException as e:
                    if task_id in task_id2positives:
                        if task_id in task_id2pairs:
                            task_id2pairs[task_id].append((choice(task_id2positives[task_id]), generation))
                        else:
                            task_id2pairs[task_id] = [(choice(task_id2positives[task_id]), generation)]
                    elif task_id in task_id2pairs:
                        task_id2pairs[task_id].append((task_id2reference[task_id], generation))
                    else:
                        task_id2pairs[task_id] = [(task_id2reference[task_id], generation)]
            else:
                if task_id in task_id2positives:
                    if task_id in task_id2pairs:
                        task_id2pairs[task_id].append((choice(task_id2positives[task_id]), sample[-1]['generation']))
                    else:
                        task_id2pairs[task_id] = [(choice(task_id2positives[task_id]), sample[-1]['generation'])]
                elif task_id in task_id2pairs:
                    task_id2pairs[task_id].append((task_id2reference[task_id], sample[-1]['generation']))
                else:
                    task_id2pairs[task_id] = [(task_id2reference[task_id], sample[-1]['generation'])]

    with open('temp.json', 'w') as f:
        dump(task_id2pairs, f, indent=4)
