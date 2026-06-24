# PM2 EcoRecover AlgaeCentrifuge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `AlgaeCentrifuge` SanUnit that behaves like a `Splitter` and adds centrifuge design, purchase cost, stainless-steel mass, and energy accounting.

**Architecture:** Implement one new class in `exposan/pm2_ecorecover_lca/_sanunits.py` and add focused tests to `tests/test_pm2_ecorecover_lca.py`. The unit will inherit `su.Splitter`, leave `_run()` inherited, and keep all new work in `_design()` and `_cost()`.

**Tech Stack:** Python, QSDsan `WasteStream` and `su.Splitter`, BioSTEAM-style centrifuge cost correlation, `pytest`.

---

## File Structure

- Modify `exposan/pm2_ecorecover_lca/_sanunits.py`
  - Export `AlgaeCentrifuge`.
  - Add `AlgaeCentrifuge` after `Ultrafiltration` and before `Photobioreactor`.
  - Include references in the class docstring and source comments.
- Modify `tests/test_pm2_ecorecover_lca.py`
  - Import `AlgaeCentrifuge`.
  - Add focused tests for splitter behavior, energy equations, purchase cost, weight, zero-flow behavior, and validation.

---

### Task 1: Add Failing Tests For Base AlgaeCentrifuge Behavior

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add import and fixture**

Update the import:

```python
from exposan.pm2_ecorecover_lca._sanunits import (
    AlgaeCentrifuge, CO2Supply, Tank, Ultrafiltration,
)
```

Add this fixture near the existing fixtures:

```python
@pytest.fixture
def algae_centrifuge():
    pc.create_pm2_cmps()
    feed = WasteStream('centrifuge_feed', H2O=10000, units='kg/hr')
    return AlgaeCentrifuge(
        'CENT_test',
        ins=feed,
        outs=('centrate', 'algae_slurry'),
        split=0.4,
    )
```

- [ ] **Step 2: Add Splitter and equation tests**

Add:

```python
def test_algae_centrifuge_inherits_splitter_run(algae_centrifuge):
    feed = algae_centrifuge.ins[0]
    algae_centrifuge.simulate()
    centrate, algae_slurry = algae_centrifuge.outs
    assert centrate.F_mass == pytest.approx(feed.F_mass * 0.4)
    assert algae_slurry.F_mass == pytest.approx(feed.F_mass * 0.6)


def test_algae_centrifuge_design_and_energy(algae_centrifuge):
    algae_centrifuge.phi = 0.1
    algae_centrifuge.simulate()
    D = algae_centrifuge.design_results

    Q_hr = algae_centrifuge.ins[0].get_total_flow('m3/hr')
    Q_s = algae_centrifuge.ins[0].get_total_flow('m3/s')
    mu = 497e-3 / (25 + 42.5)**1.5
    vg = (1050 - 1000) * (5e-6)**2 * 9.8 / (18 * mu)
    vg_eff = vg * (1 - 0.1)**4.65
    Qm = Q_s * 3600 * 0.1e-6 / vg_eff
    E_disc = 1.447 * Qm**(-0.304)
    power = E_disc * Q_hr
    weight = max(1126.1 * np.log(Q_hr) - 1204.8, 0.)

    assert D['Influent flow'] == pytest.approx(Q_hr)
    assert D['Water viscosity'] == pytest.approx(mu)
    assert D['Gravity settling velocity'] == pytest.approx(vg)
    assert D['Effective settling velocity'] == pytest.approx(vg_eff)
    assert D['Master-curve flow'] == pytest.approx(Qm)
    assert D['Disc centrifuge energy intensity'] == pytest.approx(E_disc)
    assert D['Centrifuge stainless steel'] == pytest.approx(weight)
    assert algae_centrifuge.power_utility.rate == pytest.approx(power)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_algae_centrifuge_inherits_splitter_run tests/test_pm2_ecorecover_lca.py::test_algae_centrifuge_design_and_energy -q
```

Expected: fail with `ImportError` because `AlgaeCentrifuge` is not defined yet.

---

### Task 2: Implement Core AlgaeCentrifuge Class

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Export class**

Add `AlgaeCentrifuge` to `__all__`:

```python
__all__ = ('Photobioreactor',
           'Ecorecoverypump',
           'Tank',
           'CO2Supply',
           'Ultrafiltration',
           'AlgaeCentrifuge',
           )
```

- [ ] **Step 2: Add class skeleton, docstring, and constructor**

Insert after `Ultrafiltration`:

```python
class AlgaeCentrifuge(su.Splitter):
    '''
    Splitter-based algae centrifuge with design, purchase-cost, and energy
    accounting.

    The material split is inherited from :class:`qsdsan.unit_operations.Splitter`;
    this class only adds centrifuge design and cost accounting.

    Parameters
    ----------
    split : float, array, or dict
        Split fraction passed to :class:`Splitter`; units are dimensionless.
    algal_cell_diameter : float
        Algal particle diameter in [m].
    algal_particle_density : float
        Algal particle density in [kg/m3].
    water_density : float
        Water density in [kg/m3].
    T : float
        Water temperature used in the viscosity equation in [deg C].
    phi : float
        Solids volume fraction for hindered-settling correction; units are
        dimensionless.
    vgm : float
        Master-curve settling velocity in [m/s].

    References
    ----------
    Cost follows BioSTEAM ``LiquidsCentrifuge``:
    ``C = 28100 * Q**0.574`` with ``Q`` in [m3/hr], CEPCI 525.4,
    upper-bound flow 100 m3/hr, and bare-module factor 2.03.

    Energy equations follow Najjar and Abu-Shamleh, Algal Research 51
    (2020) 102046, http://doi.org/10.1016/j.algal.2020.102046.

    Stainless-steel weight follows the Dolphin Centrifuge capacity
    correlation:
    ``weight = 1126.1*ln(Q) - 1204.8`` with ``Q`` in [m3/hr].
    '''
    _F_BM_default = {'Centrifuge': 2.03}
    _units = {
        'Influent flow': 'm3/hr',
        'Water viscosity': 'Pa*s',
        'Gravity settling velocity': 'm/s',
        'Effective settling velocity': 'm/s',
        'Master-curve flow': 'm3/hr',
        'Disc centrifuge energy intensity': 'kWh/m3',
        'Centrifuge stainless steel': 'kg',
        'Number of centrifuges': '',
    }

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *, split,
            order=None, init_with='WasteStream', F_BM_default=None,
            isdynamic=False, algal_cell_diameter=5e-6,
            algal_particle_density=1050., water_density=1000.,
            T=25., phi=0., vgm=0.1e-6,
        ):
        su.Splitter.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        self.algal_cell_diameter = algal_cell_diameter
        self.algal_particle_density = algal_particle_density
        self.water_density = water_density
        self.T = T
        self.phi = phi
        self.vgm = vgm
```

- [ ] **Step 3: Add validation and calculation helpers**

Add:

```python
    @staticmethod
    def _require_positive(name, value):
        if value <= 0:
            raise ValueError(f'`{name}` must be positive.')
        return value

    @property
    def algal_cell_diameter(self):
        return self._algal_cell_diameter

    @algal_cell_diameter.setter
    def algal_cell_diameter(self, value):
        self._algal_cell_diameter = self._require_positive(
            'algal_cell_diameter', value,
        )

    @property
    def water_density(self):
        return self._water_density

    @water_density.setter
    def water_density(self, value):
        self._water_density = self._require_positive('water_density', value)

    @property
    def algal_particle_density(self):
        return self._algal_particle_density

    @algal_particle_density.setter
    def algal_particle_density(self, value):
        if value <= self.water_density:
            raise ValueError(
                '`algal_particle_density` must be greater than '
                '`water_density`.',
            )
        self._algal_particle_density = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value <= -42.5:
            raise ValueError('`T` must be greater than -42.5 deg C.')
        self._T = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if not 0 <= value < 1:
            raise ValueError('`phi` must be greater than or equal to 0 and less than 1.')
        self._phi = value

    @property
    def vgm(self):
        return self._vgm

    @vgm.setter
    def vgm(self, value):
        self._vgm = self._require_positive('vgm', value)

    def _get_mu(self):
        # Source: same viscosity relationship used in Ultrafiltration.
        return 497e-3 / (self.T + 42.5)**1.5

    def _get_gravity_settling_velocity(self, mu):
        # Source: Eq. 7 in Najjar and Abu-Shamleh (2020).
        return (
            (self.algal_particle_density - self.water_density)
            * self.algal_cell_diameter**2 * 9.8 / (18 * mu)
        )
```

- [ ] **Step 4: Add `_design()` and `_cost()`**

Add:

```python
    def _design(self):
        D = self.design_results
        Q_hr = self.ins[0].get_total_flow('m3/hr')
        mu = self._get_mu()
        vg = self._get_gravity_settling_velocity(mu)
        vg_eff = vg * (1 - self.phi)**4.65

        if Q_hr > 0:
            # Source: Eq. 15 in Najjar and Abu-Shamleh (2020).
            Q_s = self.ins[0].get_total_flow('m3/s')
            Qm = Q_s * 3600 * self.vgm / vg_eff
            # Source: Eq. 16 in Najjar and Abu-Shamleh (2020), disc centrifuge.
            E_disc = 1.447 * Qm**(-0.304)
            # Source: Dolphin Centrifuge capacity correlation.
            weight = max(1126.1 * np.log(Q_hr) - 1204.8, 0.)
        else:
            Qm = E_disc = weight = 0.

        D['Influent flow'] = Q_hr
        D['Water viscosity'] = mu
        D['Gravity settling velocity'] = vg
        D['Effective settling velocity'] = vg_eff
        D['Master-curve flow'] = Qm
        D['Disc centrifuge energy intensity'] = E_disc
        D['Centrifuge stainless steel'] = weight
        D['Number of centrifuges'] = ceil(Q_hr / 100) if Q_hr > 0 else 0

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        Q_hr = D['Influent flow']
        N = D['Number of centrifuges']
        if Q_hr > 0:
            Q_each = Q_hr / N
            # Source: BioSTEAM LiquidsCentrifuge cost correlation.
            C['Centrifuge'] = (
                N * CEPCI_by_year[2022] / 525.4
                * 28100 * Q_each**0.574
            )
        else:
            C['Centrifuge'] = 0.
        self.power_utility.rate = (
            D['Disc centrifuge energy intensity'] * Q_hr
        )
```

- [ ] **Step 5: Run base tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_algae_centrifuge_inherits_splitter_run tests/test_pm2_ecorecover_lca.py::test_algae_centrifuge_design_and_energy -q
```

Expected: PASS.

---

### Task 3: Add Cost, Zero-Flow, And Validation Tests

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add tests**

Add:

```python
def test_algae_centrifuge_cost_and_parallel_count(algae_centrifuge):
    algae_centrifuge.ins[0].set_flow([250000], ['H2O'], units='kg/hr')
    algae_centrifuge.simulate()
    D = algae_centrifuge.design_results
    Q_hr = algae_centrifuge.ins[0].get_total_flow('m3/hr')
    N = int(np.ceil(Q_hr / 100))
    Q_each = Q_hr / N
    expected_cost = N * CEPCI_by_year[2022] / 525.4 * 28100 * Q_each**0.574

    assert D['Number of centrifuges'] == N
    assert algae_centrifuge.baseline_purchase_costs['Centrifuge'] == pytest.approx(
        expected_cost
    )
    assert algae_centrifuge.F_BM['Centrifuge'] == pytest.approx(2.03)


def test_algae_centrifuge_handles_zero_flow():
    pc.create_pm2_cmps()
    feed = WasteStream('empty_centrifuge_feed')
    centrifuge = AlgaeCentrifuge(
        'CENT_empty',
        ins=feed,
        outs=('empty_centrate', 'empty_algae_slurry'),
        split=0.4,
    )
    centrifuge.simulate()
    assert centrifuge.design_results['Influent flow'] == 0
    assert centrifuge.design_results['Master-curve flow'] == 0
    assert centrifuge.design_results['Disc centrifuge energy intensity'] == 0
    assert centrifuge.design_results['Centrifuge stainless steel'] == 0
    assert centrifuge.design_results['Number of centrifuges'] == 0
    assert centrifuge.baseline_purchase_costs['Centrifuge'] == 0
    assert centrifuge.power_utility.rate == 0


@pytest.mark.parametrize(
    ('name', 'value'),
    (
        ('algal_cell_diameter', 0),
        ('water_density', 0),
        ('algal_particle_density', 999),
        ('T', -42.5),
        ('phi', -0.1),
        ('phi', 1),
        ('vgm', 0),
    ),
)
def test_algae_centrifuge_rejects_invalid_inputs(algae_centrifuge, name, value):
    with pytest.raises(ValueError, match=name):
        setattr(algae_centrifuge, name, value)
```

- [ ] **Step 2: Run tests to verify failure or pass**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q -k algae_centrifuge
```

Expected: PASS after Task 2 implementation; if any test fails, fix the class without changing the expected equations.

---

### Task 4: Final Verification

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Modify: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Compile edited module**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m py_compile exposan/pm2_ecorecover_lca/_sanunits.py
```

Expected: command exits with code 0.

- [ ] **Step 2: Run new AlgaeCentrifuge tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q -k algae_centrifuge
```

Expected: PASS.

- [ ] **Step 3: Run all locally focused tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q
```

Expected: all tests unrelated to the pre-existing `CO2Supply` issue pass; if the same `CO2Supply` failures remain, report them as pre-existing and not caused by `AlgaeCentrifuge`.

- [ ] **Step 4: Review changed files**

Run:

```bash
git status --short
git diff -- exposan/pm2_ecorecover_lca/_sanunits.py tests/test_pm2_ecorecover_lca.py docs/superpowers/plans/2026-06-22-pm2-ecorecover-algae-centrifuge.md
```

Expected: changes are limited to `AlgaeCentrifuge`, tests, and the implementation plan, aside from already existing unrelated local edits.
