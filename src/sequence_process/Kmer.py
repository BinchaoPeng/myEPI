from itertools import product

import numpy as np
from joblib import Parallel, delayed


class Kmer:
    """
    Kmer：Kmer描述给定序列中k个相邻核酸的出现频率。
    可以将k=1，2，3，4，5，...，n等的特征向量拼接起来作为一个组合的特征向量使用,还可以设置步长step
    考虑**核酸组成(NAC)、二核酸组成(DNC)、三核酸组成(TNC)和四核酸组成(TeNC)**。
    NAC、DNC、TNC、TENC分别生成4D、16D、64D和256D特征向量。随后，将这四个合成的特征向量连接起来，得到340D的特征向量。
    format:
    [N(A)/K1,N(C)/K1,N(G)/K1,N(T)/K1,N(AA)/K2,...,N(TT)/K2,N(AAA)/K3,...]
    freq = N /K
    """

    def __init__(self, k=range(1, 5), n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs

    def get_Params(self):
        print("Kmer Params:", self.__dict__)

    # 提取核苷酸类型（排列组合）
    def nucleotide_type(self, k):
        z = []
        for i in product('ACGT', repeat=k):  # 笛卡尔积（有放回抽样排列）
            # print(i)
            z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
        # print(z)
        return z

    def _get_Kmer_item(self, seq, k_item):
        N = len(seq) - k_item + 1
        atgc_list = self.nucleotide_type(k_item)
        freq_atgc_dict = dict(zip(atgc_list, np.zeros(len(atgc_list))))
        for idx in range(0, len(seq) - k_item + 1):
            atgc = seq[idx:idx + k_item]
            # print(k_item, " | ", atgc)
            try:
                freq_atgc_dict[atgc] += 1
            except KeyError:
                N -= 1
        # print(k_item, " | ", freq_atgc_dict)
        for k, v in freq_atgc_dict.items():
            freq_atgc_dict[k] = v / N
        # print(k_item, " | ", freq_atgc_dict)
        return np.array(list(freq_atgc_dict.values()))

    def get_Kmer(self, seq):
        kmer_vec = []
        for k_item in self.k:
            kmer_item_vec = self._get_Kmer_item(seq, k_item)
            # print(kmer_item_vec)
            kmer_vec.extend(kmer_item_vec)
        return kmer_vec

    def run_Kmer(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_Kmer)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)


if __name__ == '__main__':
    kmer = Kmer()
    kmer_vec = kmer.get_Kmer("AGCTNACGT")
    print(len(kmer_vec))
    print(kmer_vec)
