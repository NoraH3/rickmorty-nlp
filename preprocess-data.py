import glob
import io
import os
import re

import numpy as np
from tokenizers import CharBPETokenizer


def get_dataset_files(transcript_folder, summary_folder, validation_size=0.2):
    transcripts = sorted(glob.glob(transcript_folder + '/*.txt'))
    summaries = []

    for file_name in transcripts:
        split = os.path.split(file_name)
        summaries.append(os.path.join(summary_folder, split[-1]))

    ds_size = len(transcripts)

    # split into training and test set
    np.random.seed(1337)
    validation_size = int(ds_size * validation_size)
    selection = np.random.choice(ds_size, validation_size, replace=False)

    train_files = []
    test_files = []
    train_result_files = []
    test_result_files = []

    for i in range(ds_size):
        if i not in selection:
            train_files.append(transcripts[i])
            train_result_files.append(summaries[i])
        else:
            test_files.append(transcripts[i])
            test_result_files.append(summaries[i])

    train_files = np.asarray(train_files)
    train_result_files = np.asarray(train_result_files)
    test_files = np.asarray(test_files)
    test_result_files = np.asarray(test_result_files)

    # random shuffle
    indices = np.arange(train_files.shape[0])
    np.random.shuffle(indices)
    train_files = train_files[indices]
    train_result_files = train_result_files[indices]

    indices = np.arange(test_files.shape[0])
    np.random.shuffle(indices)
    test_files = test_files[indices]
    test_result_files = test_result_files[indices]

    return train_files, train_result_files, test_files, test_result_files


def cleanup_text(content):
    content = re.sub(r"\[.*\]", '', content)  # remove [comments]
    content = re.sub(r"\*.*\*", '', content)  # remove *comments*
    content = re.sub(r"\’", '\'', content)  # replace special ` apostrophes
    content = re.sub(r"\‘", '\'', content)  # replace special ` apostrophes
    content = re.sub(r"\\", '-', content)  # replace / with -

    for i in range(10):
        content = re.sub(r"\n\n", '\n', content)
        content = re.sub(r"\s\s", ' ', content)

    content = re.sub(r"\n", ' ', content)
    return content


def preprocess_data(files, cleanup=False):
    data = []

    for filename in files:
        file = io.open(filename, mode="r", encoding="utf-8")
        data.append(file.read())

    if cleanup:
        for i, content in enumerate(data):
            data[i] = cleanup_text(content)
    return data


def get_dataset(train_files, train_result_files, test_files, test_result_files):
    train_data = preprocess_data(train_files, cleanup=True)
    train_results = preprocess_data(train_result_files)
    test_data = preprocess_data(test_files, cleanup=True)
    test_results = preprocess_data(test_result_files)

    return train_data, train_results, test_data, test_results


def tokenize_data(tokenizer, data):
    for i, sample in enumerate(data):
        data[i] = tokenizer.encode(sample)

    return data


def get_data():
    transcript_folder = os.path.join('data', 'transcripts')
    summary_folder = os.path.join('data', 'summary')

    train_files, train_result_files, test_files, test_result_files = get_dataset_files(transcript_folder,
                                                                                       summary_folder)
    train_data, train_results, test_data, test_results = get_dataset(train_files, train_result_files, test_files,
                                                                     test_result_files)

    tokenizer = CharBPETokenizer()
    all_files = np.concatenate([train_files, train_result_files, test_files, test_result_files])
    tokenizer.train(list(all_files))

    train_data = tokenize_data(tokenizer, train_data)
    test_data = tokenize_data(tokenizer, test_data)

    return train_data, train_results, test_data, test_results


if __name__ == "__main__":
    get_data()
