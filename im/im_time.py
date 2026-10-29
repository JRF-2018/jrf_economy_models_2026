# im_time.py: 段 17 の時間の層の道具(模型の外側の包み)。
# 粗い時間の進め方(projective integration。Kevrekidis et al. 2003、Gear and Kevrekidis 2003 の equation-free):
# 経済を短く回して、遅い量(災害の頻度の信念と守りの資本)の変わる向きを推定し、その向きに長く飛ぶ。飛ぶあいだ、経済の速い量は回さない。
import numpy as np

SLOW = ('n_eff', 't_eff', 'prot')


def slow_state(e, slow=SLOW):
    return np.array([getattr(e, k, 0.0) for k in slow], float)


def set_slow(e, v, slow=SLOW):
    for k, x in zip(slow, v):
        setattr(e, k, float(max(x, 0.0)))


def projective(e, T, inner=10, jump=40, burn=5, slow=SLOW):
    """T 年ぶんを、「inner 年回し(最初の burn 年は速い量をならすため、傾きに使わない)、遅い量を jump 年ぶん外挿して飛ぶ」の繰り返しで進める。
    戻り値: 実際に回した年数と、年ごとの (年、信念、守りの資本) の記録(飛んだ年は外挿の値)。"""
    rec, t, steps = [], 0, 0
    while t < T:
        xs = []
        for k in range(inner):
            e.step(); steps += 1; t += 1
            x = slow_state(e, SLOW)
            rec.append((t, e.belief, e.prot))
            if k >= burn:
                xs.append(x)
            if t >= T:
                return steps, rec
        xs = np.array(xs)
        slope = np.polyfit(np.arange(len(xs)), xs, 1)[0]      # 遅い量の一年あたりの変化
        J = min(jump, T - t)
        x1 = xs[-1] + slope * J
        keep = np.array([k in slow for k in SLOW])
        set_slow(e, np.where(keep, x1, xs[-1]))              # 飛ばすのは slow に入れた量だけ。ほかは飛ぶ前の値のまま
        a, b = e.p.prior_a + e.n_eff, e.p.prior_b + e.t_eff
        for j in range(1, J + 1):
            rec.append((t + j, (e.p.prior_a + xs[-1][0] + slope[0] * j) / (e.p.prior_b + xs[-1][1] + slope[1] * j),
                        xs[-1][2] + (slope[2] * j if 'prot' in slow else 0.0)))
        t += J
        e.t += J
    return steps, rec


# ---------------------------------------------------------------- 平行世界(多段の分割。段 17b)
# 稀な出来事(国債/所得が高い水準を超える、など)の確率を、「危機に近づいた世界を複製し、乱数を替えて未来を分ける」ことで推定する。
# 多段の分割(multilevel splitting。Kahn and Harris 1951、Glasserman et al. 1999 の固定の手間の版): 閾値の列 ℓ1 < ℓ2 < … < ℓK を置き、
# 段 k では N 個の世界を、ℓk を初めて超えた時点から走らせ、ℓ(k+1) に届いた割合 p_k を数える。届いた世界を N 個に複製して次の段へ。
# 推定 = p_0 × p_1 × … × p_(K−1)。重みは使わない(複製は、確率を均等に分ける)。
import copy


def reseed(e, rng):
    """世界の中のすべての乱数の発生器を、新しい種で替える(複製した世界の未来を分ける)。"""
    for k, v in list(vars(e).items()):
        if isinstance(v, np.random.Generator):
            setattr(e, k, np.random.default_rng(int(rng.integers(2 ** 62))))
    if getattr(e, 'fp', None) is not None:
        e.fp.rng = np.random.default_rng(int(rng.integers(2 ** 62)))


def run_until(e, T, level, measure, t_from):
    """世界 e を T 年まで進め、t_from 年以降に measure(rec) が level を初めて超えたら、そこで止めて (True, 年数) を返す。"""
    steps = 0
    while e.t < T:
        try:
            r = e.step(); steps += 1
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return False, steps                                # 壊れた世界(価格が解けない)は、そこで届かなかったと数える(素朴な推定と同じ扱い)
        if e.t > t_from and measure(r) > level:
            return True, steps
    return False, steps


def _stage_job(args):
    e, T, level, t_from, seed, measure_name = args
    if e.log and e.t > t_from and MEASURES[measure_name](e.log[-1]) > level:
        return e, True, 0                                  # 前の閾値を超えた年に、この閾値もすでに超えていた(飛び越し)
    rng = np.random.default_rng(seed)
    reseed(e, rng)
    hit, steps = run_until(e, T, level, MEASURES[measure_name], t_from)
    return (e if hit else None), hit, steps


def _b_fwd(r):
    return r['b'] + 5.0 * r['deficit'] / r['Y']             # 国債/所得 + 5 年ぶんの赤字/所得(近い未来を見越す、速い量を足した物差し。段 17c)


MEASURES = {'b': lambda r: r['b'], 'b_fwd': _b_fwd}


# ---------------------------------------------------------------- 適応型の平行世界(段 17c)
# 閾値を前もって決めず、毎回、N 個の世界を最後まで走らせて「届いた最高の水準」を集め、上位 p0 の境目を次の閾値にする
# (adaptive multilevel splitting。Cérou and Guyader 2007 の、一度に多くを捨てる版)。届いた世界の、閾値を初めて超えた時点の状態は、
# 同じ種で走らせ直して取り出す(世界の乱数はすべて種で決まるので、同じ道をたどる)。


def _max_job(args):
    e, T, t_from, seed, measure_name = args
    m = MEASURES[measure_name]
    best = m(e.log[-1]) if (e.log and e.t > t_from) else -np.inf
    reseed(e, np.random.default_rng(seed))
    steps = 0
    while e.t < T:
        try:
            r = e.step(); steps += 1
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            break                                              # 壊れた世界は、そこまでの最高の水準で数える
        if e.t > t_from:
            best = max(best, m(r))
    return best, steps


def adaptive(make, target, T, N, p0=0.3, seed=0, t_from=100, measure='b', pool=None, max_iter=40):
    """戻り値: 推定、使った閾値の列、回した年数の和。目当て(measure > target)に届いた割合が p0 を超えたら、その割合を掛けて終わる。"""
    rng = np.random.default_rng(seed)
    worlds = [make(i) for i in range(N)]
    est, levels, total = 1.0, [], 0
    for _ in range(max_iter):
        seeds = [int(rng.integers(2 ** 62)) for _ in worlds]
        res = list(pool.map(_max_job, [(w, T, t_from, sd, measure) for w, sd in zip(worlds, seeds)])) if pool else \
            [_max_job((w, T, t_from, sd, measure)) for w, sd in zip(worlds, seeds)]
        best = np.array([r[0] for r in res])
        total += sum(r[1] for r in res)
        frac = float((best > target).mean())
        lv = float(np.quantile(best, 1 - p0))
        if frac >= p0 or lv >= target:
            return est * frac, levels, total
        surv = np.nonzero(best > lv)[0]
        if len(surv) == 0:
            return 0.0, levels, total
        est *= len(surv) / N
        levels.append(lv)
        # 届いた世界を、同じ種で閾値を初めて超えた時点まで走らせ直す
        jobs = [(worlds[i], T, lv, t_from, seeds[i], measure) for i in surv]
        rep = list(pool.map(_stage_job, jobs)) if pool else [_stage_job(j) for j in jobs]
        total += sum(r[2] for r in rep)
        hits = [r[0] for r in rep if r[1]]
        if not hits:
            return 0.0, levels, total
        idx = rng.integers(len(hits), size=N)
        worlds = [copy.deepcopy(hits[i]) for i in idx]
    return est * frac, levels, total


def splitting(make, levels, T, N, seed=0, t_from=100, measure='b', pool=None, measure_final=None):
    """make(i): i 番目の初めの世界を作る。levels: 閾値の列(最後が目当ての出来事)。戻り値: 推定、段ごとの割合、回した年数の和。
    measure_final: 最後の段(目当ての出来事)だけ別の物差しで数える(途中の段は measure。段 17c)。"""
    rng = np.random.default_rng(seed)
    worlds = [make(i) for i in range(N)]
    ps, total = [], 0
    for k, lv in enumerate(levels):
        m = (measure_final or measure) if k == len(levels) - 1 else measure
        jobs = [(w, T, lv, t_from, int(rng.integers(2 ** 62)), m) for w in worlds]
        res = list(pool.map(_stage_job, jobs)) if pool else [_stage_job(j) for j in jobs]
        total += sum(r[2] for r in res)
        hits = [r[0] for r in res if r[1]]
        ps.append(len(hits) / N)
        if not hits:
            return 0.0, ps, total
        if k < len(levels) - 1:
            idx = rng.integers(len(hits), size=N)            # 届いた世界を N 個に複製する(平行世界)
            worlds = [copy.deepcopy(hits[i]) for i in idx]
    return float(np.prod(ps)), ps, total
