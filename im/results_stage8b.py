# results_stage8b.py: 段 8b 文化から技術への波及。python -W ignore results_stage8b.py > results_stage8b.txt
import results_stage8 as r8
from results_stage5b import tail

R = dict(research=True, s_R=0.02, **tail(1.5))
cols = [('雇用', 'E', '.0f'), ('技術の進み', 'prod', '.4f'), ('種類の数', 'n_var', '.2f'), ('贅沢品の実効の量', 'lux_eff', '.0f'), ('所得の上位10%', 'top10i', '.3f')]

if __name__ == '__main__':
    r8.table('V8b 文化から技術への波及(研究の当たる確率 × 種類の数^波及)。研究 2%、ψ = 1、ε = 2。倍率の上限 3(既定)',
             (('文化なし(ψ=0)', R), ('文化、波及なし', dict(R, psi_v=1.0, eps_v=2.0)), ('文化、波及 0.5', dict(R, psi_v=1.0, eps_v=2.0, culture_spill=0.5)),
              ('文化、波及 1.0', dict(R, psi_v=1.0, eps_v=2.0, culture_spill=1.0)),
              ('文化、波及 1.0、上限 1.5', dict(R, psi_v=1.0, eps_v=2.0, culture_spill=1.0, spill_cap=1.5))), cols=cols)
    # 上限なし(冪のまま)の波及: 当たり → 知財 → 種類 → 当たりの帰還で発散するか
    ok = [r8.one((dict(R, psi_v=1.0, eps_v=2.0, culture_spill=s, spill_cap=1e9), seed, False)) is not None for s in (0.5, 1.0) for seed in r8.SEEDS]
    print(f'  上限なしの波及(0.5、1.0): 最後まで走れた種 {sum(ok[:12])}/12、{sum(ok[12:])}/12(残りは発散して止まった)')
