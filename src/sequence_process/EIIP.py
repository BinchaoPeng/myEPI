from itertools import product

import numpy as np
from joblib import Parallel, delayed


class EIIP:
    """
    显示自由电子能量分布，{A，C，G，T}值为{0.1260，0.1340，0.0806，0.1335}。
    S = [EIIP~AAA~f~AAA~, ... , EIIP~TTT~f~TTT~]
    其中EIIP~pqr~=EIIP~p~+EIIP~q~+EIIP~r~表示三核苷酸(pqr)EIIP值之一，并且pqr∈{G，A，C，T}； f~pqr~表示三核苷酸(pqr)的频率。
    最终，EIIP提供了64D特征向量。
    """
    EIIP_value_dict = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335}

    def __init__(self, k=range(1, 5), n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs

    # 提取核苷酸类型（排列组合）
    def nucleotide_type(self, k):
        z = []
        for i in product('ACGT', repeat=k):  # 笛卡尔积（有放回抽样排列）
            # print(i)
            z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
        # print(z)
        return z

    def _get_EIIP_freq(self, seq, k_item):
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

    def _get_EIIP_value(self, k_item):
        EIIP_dict = {}

        for i in product('ACGT', repeat=k_item):
            v = 0
            for s in i:
                v += self.EIIP_value_dict[s]
            k = ''.join(i)  # 把('A,A,A')转变成（AAA）形式
            EIIP_dict.update({k: v})
        # print(EIIP_dict)
        return np.array(list(EIIP_dict.values()))

    def get_EIIP(self, seq):
        feature = []
        for k_item in self.k:
            EIIP_freq = self._get_EIIP_freq(seq, k_item)
            EIIP_value = self._get_EIIP_value(k_item)
            # print(EIIP_freq)
            # print(EIIP_value)
            feature_item = [freq * value for freq, value in zip(EIIP_freq, EIIP_value)]
            # print(feature_item)
            feature.extend(feature_item)
        return np.array(feature)

    def run_EIIP(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_EIIP)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)


if __name__ == '__main__':
    eiip = EIIP()
    feature = eiip.get_EIIP("AGTCACGTN")
    print(len(feature))
