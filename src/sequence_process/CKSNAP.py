import numpy as np
from joblib import Parallel, delayed


class CKSNAP:
    """
    组成的核苷酸对("GC"，"CC"，"AT"，"AA"，"AG"，"AC"，"GA"，"CT"，"CA"，"GG"，"GT"，"CG"，"TG"，"TG"，"TT"，"TA")是由k个间隔的两个核苷酸组成的
    计算被任意k个核酸分隔的核苷酸bp的出现频率。即计算两个核苷酸分别位于`i`和`i + K + 1`位置的核苷酸对的频率。
    format:
    (F_GC/N,F_CC/N,...,F_TA/N)
    F_XX：frequency of 核苷酸对XX
    """

    def __init__(self, seq="ATGCACGCAT", K=[x for x in range(0, 6)], n_jobs=3):
        self.n_jobs = n_jobs
        self.seq = seq
        self.K = K
        self.ATGC_base = {"AA": 0, "AC": 0, "AG": 0, "AT": 0,
                          "CA": 0, "CC": 0, "CG": 0, "CT": 0,
                          "GA": 0, "GC": 0, "GG": 0, "GT": 0,
                          "TA": 0, "TC": 0, "TG": 0, "TT": 0}

    def get_CKSNAP(self, k):
        ATGC = self.ATGC_base.copy()
        N = len(self.seq) - k - 1
        for idx in range(0, N):
            atgc = self.seq[idx] + self.seq[idx + k + 1]
            # print(atgc)
            if atgc.__contains__("N"):
                N = N - 1
            else:
                ATGC[atgc] += 1
        F_atgc = [v / N for v in ATGC.values()]
        # print(F_atgc)
        return F_atgc

    def run_CKSNAP(self):
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_CKSNAP)
                           (k=k)
                           for k in self.K
                           )
            all_out.extend(out)
        return np.array(all_out).flatten()


if __name__ == '__main__':
    cksnap = CKSNAP("ATGCATGCAGGC")
    all_out = cksnap.run_CKSNAP()
    print(all_out)
    print(len(all_out))
