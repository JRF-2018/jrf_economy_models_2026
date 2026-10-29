# portable.py: 持ち運べる乱数と解き方(ブラウザの版の段 B0)
# ブラウザの版(JS)と同じ数を出すための、両方の言語で同じ手順で書ける乱数の発生器と、Levenberg–Marquardt の最小化。
# im の rng='portable' と solver='portable' のときに使う(既定は numpy の乱数と scipy の least_squares のまま)。
# JS の写しは site/js/model/rng.js と solver.js。手順を変えるときは、両方を同じに直す。
import math
import numpy as np

M32 = 0xFFFFFFFF


def _splitmix32(x):
    """32 ビットの splitmix。種から発生器の状態を作る。戻り値は (次の x, 出力)。"""
    x = (x + 0x9E3779B9) & M32
    z = x
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & M32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & M32
    z = z ^ (z >> 16)
    return x, z


def _rotl(x, k):
    return ((x << k) | (x >> (32 - k))) & M32


class PortableRNG:
    """xoshiro128** の発生器(32 ビットの演算だけで書けるので、JS でも同じ列になる)。
    numpy の Generator のうち、im が使う引き方だけを、同じ名前で持つ。引き方は逆関数法と Box–Muller で、
    numpy とは別の列になる(numpy の列をまねることはしない)。"""

    def __init__(self, seed=0):
        x = int(seed) & M32
        s = []
        for _ in range(4):
            x, z = _splitmix32(x)
            s.append(z)
        if not any(s):
            s[0] = 1
        self.s = s

    def get_state(self):
        return list(self.s)

    def set_state(self, s):
        self.s = list(s)

    def next32(self):
        s0, s1, s2, s3 = self.s
        result = (_rotl((s1 * 5) & M32, 7) * 9) & M32
        t = (s1 << 9) & M32
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 11)
        self.s = [s0, s1, s2, s3]
        return result

    def _u(self):
        a = self.next32() >> 5
        b = self.next32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0     # [0, 1) の 53 ビット

    def _many(self, f, size):
        if size is None:
            return f()
        n = int(np.prod(size))
        return np.array([f() for _ in range(n)], float).reshape(size)

    def random(self, size=None):
        return self._many(self._u, size)

    def integers(self, high):
        """0 以上 high 未満の整数(一つ)。"""
        return int(math.floor(self._u() * high))

    def _normal(self):
        u1, u2 = self._u(), self._u()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def standard_normal(self, size=None):
        return self._many(self._normal, size)

    def exponential(self, scale=1.0, size=None):
        return self._many(lambda: -math.log(1.0 - self._u()) * scale, size)

    def pareto(self, a, size=None):
        """numpy の pareto と同じ分布(ロマックス: (1 − u)^(−1/a) − 1)。"""
        return self._many(lambda: (1.0 - self._u()) ** (-1.0 / a) - 1.0, size)

    def standard_t(self, nu, size=None):
        k = int(round(nu))
        assert abs(nu - k) < 1e-12 and k >= 1, '持ち運べる t 分布は、整数の自由度だけ'

        def one():
            z = self._normal()
            c = sum(self._normal() ** 2 for _ in range(k))
            return z / math.sqrt(c / k)
        return self._many(one, size)

    def poisson(self, lam):
        """ポアソン(一つ)。lam < 30 はクヌースの掛け算、それより大きければ正規の近似(四捨五入、0 以上)。"""
        if lam <= 0:
            return 0
        if lam < 30:
            L, k, p = math.exp(-lam), 0, 1.0
            while True:
                p *= self._u()
                if p <= L:
                    return k
                k += 1
        return max(int(math.floor(lam + math.sqrt(lam) * self._normal() + 0.5)), 0)

    def permutation(self, n):
        """フィッシャー–イェーツ。"""
        a = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.integers(i + 1)
            a[i], a[j] = a[j], a[i]
        return np.array(a, int)

    def choice(self, n, p=None):
        """一つを選ぶ(p は確率。和は 1)。"""
        if p is None:
            return self.integers(n)
        u = self._u()
        c = 0.0
        for i in range(n):
            c += float(p[i])
            if u < c:
                return i
        return n - 1


def make_rng(kind, seed):
    """'numpy' なら numpy の Generator、'portable' なら PortableRNG。"""
    if kind == 'portable':
        return PortableRNG(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- Levenberg–Marquardt
class LMResult:
    def __init__(self, x, fun, nfev, nit):
        self.x, self.fun, self.nfev, self.nit = x, fun, nfev, nit


def _solve_linear(A, b):
    """ガウスの消去法(部分ピボット)。A は n × n のリストのリスト(壊す)。"""
    n = len(b)
    for k in range(n):
        piv, best = k, abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > best:
                piv, best = i, abs(A[i][k])
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            b[k], b[piv] = b[piv], b[k]
        d = A[k][k]
        if d == 0.0:
            d = 1e-300
        for i in range(k + 1, n):
            f = A[i][k] / d
            if f != 0.0:
                Ai, Ak = A[i], A[k]
                for j in range(k, n):
                    Ai[j] -= f * Ak[j]
                b[i] -= f * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        Ai = A[i]
        for j in range(i + 1, n):
            s -= Ai[j] * x[j]
        d = Ai[i] if Ai[i] != 0.0 else 1e-300
        x[i] = s / d
    return x


def _sumsq(r):
    s = 0.0
    for v in r:
        s += v * v
    return s


def lm_solve(fun, z0, xtol=1e-13, ftol=1e-13, max_iter=200):
    """残差 fun(z)(長さ m のベクトル)の二乗和を最小にする z を、Levenberg–Marquardt で求める。
    ヤコビアンは前進差分(刻み h_i = √ε × max(1, |z_i|))。減衰は A + λ diag(A)(diag は 1e-12 で下から抑える)、
    λ は良くなれば 1/3 倍、悪ければ 4 倍。止め方: 歩みの最大 ≤ xtol × (1 + 最大の |z|)、または二乗和の減りが ftol × 二乗和 以下。
    手順は JS の solver.js と同じ(演算の順も同じに書く)。"""
    z = [float(v) for v in z0]
    n = len(z)
    r = [float(v) for v in fun(np.array(z))]
    m = len(r)
    cost = _sumsq(r)
    nfev = 1
    lam = -1.0
    eps = 1.4901161193847656e-08
    nit = 0
    for nit in range(1, max_iter + 1):
        J = [[0.0] * n for _ in range(m)]
        for i in range(n):
            h = eps * max(1.0, abs(z[i]))
            zz = list(z)
            zz[i] = z[i] + h
            hh = zz[i] - z[i]
            ri = fun(np.array(zz))
            nfev += 1
            for k in range(m):
                J[k][i] = (float(ri[k]) - r[k]) / hh
        A = [[0.0] * n for _ in range(n)]
        g = [0.0] * n
        for i in range(n):
            for j in range(i, n):
                s = 0.0
                for k in range(m):
                    s += J[k][i] * J[k][j]
                A[i][j] = s
                A[j][i] = s
            s = 0.0
            for k in range(m):
                s += J[k][i] * r[k]
            g[i] = s
        if lam < 0:
            lam = 1e-3
        accepted = False
        done = False
        for _ in range(60):
            M = [list(A[i]) for i in range(n)]
            for i in range(n):
                M[i][i] = A[i][i] + lam * max(A[i][i], 1e-12)
            dz = _solve_linear(M, [-v for v in g])
            zn = [z[i] + dz[i] for i in range(n)]
            rn = [float(v) for v in fun(np.array(zn))]
            nfev += 1
            cn = _sumsq(rn)
            if cn == cn and cn <= cost:          # 数でない(nan)なら退ける
                step = 0.0
                zmax = 0.0
                for i in range(n):
                    if abs(dz[i]) > step:
                        step = abs(dz[i])
                    if abs(z[i]) > zmax:
                        zmax = abs(z[i])
                small_f = (cost - cn) <= ftol * cost
                z, r = zn, rn
                cost = cn
                lam = lam / 3.0
                accepted = True
                done = step <= xtol * (1.0 + zmax) or small_f or cost == 0.0
                break
            lam = lam * 4.0
        if not accepted or done:
            break
    return LMResult(np.array(z), np.array(r), nfev, nit)
