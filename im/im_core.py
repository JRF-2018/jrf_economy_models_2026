# im_core.py: 統合模型 im の骨組み(段 0)
# 帳簿(部門の貸借の表と取引の流れの表)、パラメータの登録簿(政策・行動・構造)、時間の層の予定表。
# お金は銀行の預金に統一する(JRF、2026-10-27)。すべての支払いは、払う側の預金を減らし、受け取る側の預金を増やす。
from dataclasses import dataclass, field
from collections import defaultdict


def _float_dict():
    return defaultdict(float)          # 平行世界(段 17b)で世界を別のプロセスへ送れるよう、ラムダを使わない
import numpy as np

KINDS = ('policy', 'behavior', 'structure')
LAYERS = ('fast', 'base', 'slow')


# ---------------------------------------------------------------- パラメータの登録簿
@dataclass
class Param:
    name: str
    value: float
    kind: str           # 'policy'(政策のつまみ)、'behavior'(行動の仮定)、'structure'(構造)
    doc: str = ''


class Params:
    """パラメータの登録簿。値は属性として読める(p.tau)。種類ごとの一覧を出せる。"""

    def __init__(self):
        object.__setattr__(self, '_reg', {})

    def add(self, name, value, kind, doc=''):
        assert kind in KINDS, kind
        assert name not in self._reg, f'{name} は登録済み'
        self._reg[name] = Param(name, value, kind, doc)
        return self

    def __getattr__(self, name):
        if name.startswith('__') or name == '_reg':
            raise AttributeError(name)                  # 別のプロセスへ送るとき(pickle)に、_reg がまだないので
        try:
            return self._reg[name].value
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name not in self._reg:
            raise AttributeError(f'{name} は登録されていない(add で登録する)')
        self._reg[name].value = value

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def copy(self, **kw):
        q = Params()
        for n, prm in self._reg.items():
            q.add(n, prm.value, prm.kind, prm.doc)
        return q.set(**kw)

    def of_kind(self, kind):
        return {n: prm.value for n, prm in self._reg.items() if prm.kind == kind}

    def table(self):
        rows = []
        for kind in KINDS:
            for n, prm in self._reg.items():
                if prm.kind == kind:
                    rows.append(f'{kind:9s} {n:10s} = {prm.value!r:10}  {prm.doc}')
        return '\n'.join(rows)


# ---------------------------------------------------------------- 帳簿
class Ledger:
    """部門の貸借の表(ストック)と、取引の流れの表(一期のフロー)。
    ストックは、持ち主から見た符号で持つ(資産は正、負債は負)。金融の手段(預金・貸出・国債・持分)は、
    部門の和がいつも 0 になる。実物の資産(資本など)は financial=False で登録し、和の検査から外す。
    bank_of を与えると(部門 → その部門が預金を持つ銀行)、銀行が複数ある経済になる。銀行をまたぐ支払いは、
    払う側の銀行から受け取る側の銀行への貸し(手段 'interbank'。ユーロ圏の TARGET2 の残高にあたる)で決済する。"""

    def __init__(self, sectors, instruments, real=(), bank_of=None):
        self.sectors = list(sectors)
        self.bank_of = dict(bank_of) if bank_of else None
        if self.bank_of and 'interbank' not in instruments:
            instruments = list(instruments) + ['interbank']
        self.instruments = list(instruments) + [r for r in real if r not in instruments]
        self.real = set(real)
        self.stock = {(s, i): 0.0 for s in self.sectors for i in self.instruments}
        self.t = 0
        self._begin()

    def _begin(self):
        self.flows = defaultdict(_float_dict)    # 流れの名前 → 部門 → 額(受け取りは正、支払いは負)
        self.fin = defaultdict(_float_dict)      # 手段 → 部門 → 取引による増減
        self.reval = defaultdict(_float_dict)    # 手段 → 部門 → 評価の変化(取引でない増減)
        self.open = dict(self.stock)

    # -- 取引
    def pay(self, name, frm, to, amount):
        """経常の支払い(賃金、消費、利子、税、移転など)。預金で決済する。"""
        if amount == 0:
            return
        self.flows[name][frm] -= amount
        self.flows[name][to] += amount
        self._move('deposits', frm, -amount)
        self._move('deposits', to, +amount)
        self._settle(frm, to, amount)

    def trade(self, instrument, buyer, seller, amount):
        """金融の取引: buyer が instrument を amount だけ得て、seller が同じだけ手放す(または発行する)。預金で決済する。
        例: 銀行が企業に貸す = trade('loans', 'bank', 'firm', L)。家計が国債を買う = trade('bonds', 'hh', 'gov', B)。"""
        if amount == 0:
            return
        self._move(instrument, buyer, +amount)
        self._move(instrument, seller, -amount)
        self._move('deposits', buyer, -amount)
        self._move('deposits', seller, +amount)
        self._settle(buyer, seller, amount)

    def _settle(self, frm, to, amount):
        """銀行をまたぐ預金の移動の決済。払う側の銀行は預金の負債が減り、受け取る側の銀行に借りができる。"""
        if not self.bank_of:
            return
        bf, bt = self.bank_of[frm], self.bank_of[to]
        if bf == bt:
            return
        # 預金の記録は「払う側の銀行の預金」として動かした。それを受け取る側の銀行に付け替える
        self._move('deposits', bf, +amount)
        self._move('deposits', bt, -amount)
        self._move('interbank', bf, -amount)
        self._move('interbank', bt, +amount)

    def produce_real(self, instrument, sector, amount):
        """実物の資産の増減(投資による資本の増加、減耗など)。金融の帳簿には効かない。"""
        assert instrument in self.real
        self.stock[(sector, instrument)] += amount
        self.fin[instrument][sector] += amount

    def revalue(self, instrument, sector, amount):
        """評価の変化(値上がり、貸倒れの棒引きなど)。取引ではないので、流れの表には出ない。"""
        self.stock[(sector, instrument)] += amount
        self.reval[instrument][sector] += amount

    def _move(self, instrument, sector, amount):
        self.stock[(sector, instrument)] += amount
        self.fin[instrument][sector] += amount

    # -- 検査
    def check(self, tol=1e-9):
        """帳簿の恒等式を確かめ、最大のずれを返す。
        (1) 経常の流れの各行は、部門の和が 0(すべての支払いに受け取り手がいる)。
        (2) 金融の手段の各列は、ストックの部門の和が 0。
        (3) ストックとフローの整合: 期末 = 期首 + 取引 + 評価の変化。
        (4) 部門ごとの予算: 経常の収支の和 = 金融の資産の取引による純増(預金を含む)。"""
        err = {}
        err['flow_rows'] = max([abs(sum(row.values())) for row in self.flows.values()], default=0.0)
        fin_instr = [i for i in self.instruments if i not in self.real]
        err['stock_cols'] = max(abs(sum(self.stock[(s, i)] for s in self.sectors)) for i in fin_instr)
        err['stock_flow'] = max(abs(self.stock[(s, i)] - self.open[(s, i)] - self.fin[i][s] - self.reval[i][s])
                                for s in self.sectors for i in self.instruments)
        bud = 0.0
        for s in self.sectors:
            income = sum(row[s] for row in self.flows.values())
            nafa = sum(self.fin[i][s] for i in fin_instr)
            bud = max(bud, abs(income - nafa))
        err['budget'] = bud
        if self.bank_of:
            # 銀行ごと: その銀行の預金の負債 = その銀行に預金を持つ部門の預金の和
            gap = 0.0
            for b in set(self.bank_of.values()):
                held = sum(self.stock[(s, 'deposits')] for s in self.sectors if self.bank_of[s] == b and s != b)
                gap = max(gap, abs(held + self.stock[(b, 'deposits')]))
            err['per_bank'] = gap
        err['max'] = max(err.values())
        if tol is not None:
            assert err['max'] < tol, err
        return err

    def close(self):
        """期を閉じる(時計を一つ進める)。帳簿の時計は基本の層の一つだけ。"""
        self.t += 1
        self._begin()

    def income(self, sector):
        return sum(row[sector] for row in self.flows.values())

    def flow_table(self):
        names = list(self.flows)
        return names, np.array([[self.flows[n][s] for s in self.sectors] for n in names])

    def balance_table(self):
        return np.array([[self.stock[(s, i)] for s in self.sectors] for i in self.instruments])


# ---------------------------------------------------------------- 部品と時間の層
@dataclass
class Component:
    """模型の部品。layer は 'fast'(年の中の釣り合いとして、base の部品の中で解く)、'base'(毎年)、
    'slow'(every 年ごと)。step(model) は、帳簿に取引を書き込む。off なら何もしない(前の段と同じ数になる)。"""
    name: str
    step: callable
    layer: str = 'base'
    every: int = 1
    on: bool = True

    def __post_init__(self):
        assert self.layer in LAYERS


class Model:
    """部品を順に動かす予定表。一年ごとに、base の部品を順に、slow の部品を every 年ごとに動かし、帳簿を検査して閉じる。"""

    def __init__(self, ledger, params, components, state=None):
        self.L = ledger
        self.p = params
        self.components = list(components)
        self.state = state if state is not None else {}
        self.history = []
        self.log = []

    def due(self, c):
        return c.on and (c.layer == 'base' or (c.layer == 'slow' and self.L.t % c.every == 0))

    def step(self, check=True, tol=1e-9):
        rec = {'t': self.L.t}
        for c in self.components:
            if self.due(c):
                out = c.step(self)
                self.log.append((self.L.t, c.name))
                if out:
                    rec.update(out)
        if check:
            rec['books'] = self.L.check(tol)['max']
        self.history.append(rec)
        self.L.close()
        return rec

    def run(self, T, **kw):
        for _ in range(T):
            self.step(**kw)
        return self.history
