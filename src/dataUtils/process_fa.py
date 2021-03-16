import pandas as pd
import numpy as np


def read_from_file_with_enter(filename):
    fr = open(filename, 'r')

    sample = ""
    samples = []
    for line in fr:
        if line.startswith('>'):
            sample = ""
            continue
        else:
            sample += line[:-1]
            samples.append(sample)
            # print(line)
            # print(sample)
    return samples


def statistics_length(samples):
    lengths = []
    for sample in samples:
        lengths.append(len(sample))
        # print(len(sample))
    return pd.DataFrame(lengths, columns=['Length'])


def fa2npy(file_src, file_tgt):
    samples = read_from_file_with_enter(file_src)
    arr = np.array(samples)
    with open(file_tgt, 'wb') as f:
        np.save(f, arr)
        print("file trans successfully")
    # np.save(file_tgt, arr)


if __name__ == '__main__':
    filePath = r"../../data/enhancer_of_HeLa-S3.fa"
    fa2npy(filePath, "../../data/enhancer_of_HeLa-S3.npy")

    filePath = r"../../data/enhancer_of_K562.fa"
    fa2npy(filePath, "../../data/enhancer_of_K562.npy")

    filePath = r"../../data/promoter_of_HeLa-S3.fa"
    fa2npy(filePath, "../../data/promoter_of_HeLa-S3.npy")

    filePath = r"../../data/promoter_of_K562.fa"
    fa2npy(filePath, "../../data/promoter_of_K562.npy")
