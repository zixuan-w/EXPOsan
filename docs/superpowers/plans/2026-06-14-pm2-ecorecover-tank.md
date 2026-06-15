# PM2 EcoRecover Tank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dynamic `Tank(CSTR)` sanitation unit with rectangular-tank design results, BioSTEAM `MixTank` equipment cost, and independently selectable aeration and mechanical-mixing electricity.

**Architecture:** Keep process dynamics entirely in the inherited QSDsan `CSTR`. Add only design, purchase-cost, validation, and utility accounting in the EXPOsan subclass. Reuse BioSTEAM's public tank cost data/functions and QSDsan's public blower helper instead of duplicating their equations.

**Tech Stack:** Python, QSDsan dynamic `CSTR`, BioSTEAM tank cost algorithms, pytest.

---

### Amendment: Default aeration airflow

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`

- [ ] **Step 1: Replace the missing-airflow error test with a fallback test**

```python
def test_tank_defaults_airflow_from_tank_volume(tank):
    tank.include_aeration_power = True
    tank.simulate()
    expected_Q_air = 0.1 * tank.V_max * 1440
    assert tank.design_results['Aeration power'] == pytest.approx(
        get_P_blower(expected_Q_air / 1440)
    )
```

- [ ] **Step 2: Run the fallback test and verify RED**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py::test_tank_defaults_airflow_from_tank_volume -q
```

Expected: FAIL because `_get_aeration_power()` still raises `ValueError` when
neither explicit `Q_air` nor `DiffusedAeration.Q_air` is available.

- [ ] **Step 3: Implement the default airflow**

After checking explicit and `DiffusedAeration` airflow, add:

```python
if Q_air is None:
    Q_air = 0.1 * self.V_max * 1440
```

Keep the existing conversion from `m3/d` to `m3/min` when calling
`get_P_blower`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Run the Tank and regression tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py tests/test_pm2.py \
  tests/test_module_conventions.py -q
```

Expected: all tests pass.

### Task 1: Add focused Tank tests

**Files:**
- Create: `tests/test_pm2_ecorecover_lca.py`
- Read: `exposan/pm2_ecorecover_lca/_sanunits.py`

- [ ] **Step 1: Write a fixture that creates PM2 components and a nonreactive Tank**

```python
import numpy as np
import pytest

from qsdsan import WasteStream
from qsdsan import process_models as pc
from qsdsan.utils import get_P_blower

from exposan.pm2_ecorecover_lca._sanunits import Tank


@pytest.fixture
def tank():
    pc.create_pm2_cmps()
    feed = WasteStream('tank_feed', H2O=1000, units='kg/hr')
    return Tank(
        'T_test',
        ins=feed,
        outs='tank_effluent',
        V_max=100,
        W_tank=5,
        D_tank=4,
        freeboard=0.5,
        t_wall=0.3,
        t_slab=0.4,
        aeration=None,
        suspended_growth_model=None,
        include_aeration_power=False,
        include_mixing_power=False,
    )
```

- [ ] **Step 2: Test rectangular geometry**

```python
def test_tank_design_geometry(tank):
    tank.simulate()
    D = tank.design_results
    assert D['Tank volume'] == pytest.approx(100)
    assert D['Tank length'] == pytest.approx(5)
    assert D['Volume of concrete wall'] == pytest.approx(
        2 * ((5 + 2 * 0.3) * 0.3 * 4.5) + 2 * (5 * 0.3 * 4.5)
    )
    assert D['Volume of concrete slab'] == pytest.approx(
        (5 + 2 * 0.3) * (5 + 2 * 0.3) * 0.4
    )
    assert D['Total volume'] == pytest.approx(125)
```

- [ ] **Step 3: Test BioSTEAM MixTank cost**

```python
def test_tank_uses_biosteam_mix_tank_cost(tank):
    tank.simulate()
    assert tank.parallel['self'] >= 1
    assert tank.baseline_purchase_costs['Tank'] > 0
```

- [ ] **Step 4: Test specific-power and velocity-gradient mixing**

```python
def test_tank_specific_mixing_power(tank):
    tank.include_mixing_power = True
    tank.kW_per_m3 = 0.01
    tank.simulate()
    assert tank.design_results['Mechanical mixing power'] == pytest.approx(1)
    assert tank.power_utility.rate == pytest.approx(1)


def test_tank_velocity_gradient_mixing_power(tank):
    tank.include_mixing_power = True
    tank.mixing_intensity = 100
    tank.simulate()
    expected = tank._mixed.mu * 100**2 / 1000 * tank.V_max
    assert tank.design_results['Mechanical mixing power'] == pytest.approx(expected)
    assert tank.power_utility.rate == pytest.approx(expected)
```

- [ ] **Step 5: Test explicit and DiffusedAeration airflow**

```python
def test_tank_explicit_airflow_power(tank):
    tank.include_aeration_power = True
    tank.Q_air = 1440
    tank.simulate()
    expected = get_P_blower(1)
    assert tank.design_results['Aeration power'] == pytest.approx(expected)
    assert tank.power_utility.rate == pytest.approx(expected)


def test_tank_diffused_aeration_airflow_power(tank):
    aeration = pc.DiffusedAeration(
        'tank_aeration', 'S_O2', V=tank.V_max, Q_air=2880
    )
    tank.aeration = aeration
    tank.Q_air = None
    tank.include_aeration_power = True
    tank.simulate()
    assert tank.design_results['Aeration power'] == pytest.approx(get_P_blower(2))
```

- [ ] **Step 6: Test combined, disabled, and invalid aeration power**

```python
def test_tank_combines_selected_power_terms(tank):
    tank.include_aeration_power = True
    tank.include_mixing_power = True
    tank.Q_air = 1440
    tank.kW_per_m3 = 0.01
    tank.simulate()
    assert tank.power_utility.rate == pytest.approx(get_P_blower(1) + 1)


def test_tank_can_disable_both_power_terms(tank):
    tank.simulate()
    assert tank.power_utility.rate == 0


def test_tank_requires_airflow_for_aeration_power(tank):
    tank.include_aeration_power = True
    with pytest.raises(ValueError, match='Q_air'):
        tank.simulate()
```

- [ ] **Step 7: Run the new tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q
```

Expected: collection fails because `Tank` is not yet defined/exported.

### Task 2: Implement Tank initialization and validation

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add imports and export**

Import `Stream`, `DiffusedAeration`, `get_P_blower`,
`compute_number_of_tanks_and_purchase_cost`, `mix_tank_purchase_cost_algorithms`,
and `vessel_material_factors`. Add `'Tank'` to `__all__` and remove the empty
export string without changing the user's other current edits.

- [ ] **Step 2: Add the CSTR-compatible constructor**

Implement `Tank.__init__` by forwarding all dynamic arguments once to
`CSTR.__init__`. Store geometry, cost, blower, and mixing parameters. Do not
call `SanUnit.__init__` a second time.

- [ ] **Step 3: Add validated properties**

Add properties for `W_tank`, `D_tank`, `freeboard`, `t_wall`, `t_slab`,
`V_wf`, `Q_air`, `mixing_intensity`, `kW_per_m3`, `vessel_type`, and
`vessel_material`. Use corrected SI defaults for the commented CSTR thickness
rule: 12 inches minimum wall thickness plus 1 inch per foot of depth above
12 feet, and slab thickness equal to wall thickness plus 2 inches.

- [ ] **Step 4: Run validation-focused tests**

Run:

```bash
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q
```

Expected: tests now collect; design/cost tests remain failing because methods
are not implemented.

### Task 3: Implement design and BioSTEAM cost

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Implement `_design()`**

Populate:

```python
D['Tank volume'] = self.V_max
D['Tank width'] = self.W_tank
D['Tank depth'] = self.D_tank
D['Tank length'] = self.V_max / self.W_tank / self.D_tank
D['Volume of concrete wall'] = ...
D['Volume of concrete slab'] = ...
D['Total volume'] = self.V_max / self.V_wf
```

Add a source comment identifying QSDsan dynamic `CSTR`'s commented design.

- [ ] **Step 2: Implement MixTank purchase cost**

Use:

```python
N, Cp = compute_number_of_tanks_and_purchase_cost(
    D['Total volume'], self.purchase_cost_algorithm
)
self.parallel['self'] = N
C['Tank'] = Cp / vessel_material_factors.get(
    self.purchase_cost_algorithm.material, 1.
)
```

Add a source comment identifying BioSTEAM `Tank._cost`/`MixTank`.

- [ ] **Step 3: Run geometry and cost tests**

Run:

```bash
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py::test_tank_design_geometry \
  tests/test_pm2_ecorecover_lca.py::test_tank_uses_biosteam_mix_tank_cost -q
```

Expected: both pass.

### Task 4: Implement aeration and mechanical-mixing power

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Implement mechanical mixing**

When enabled, calculate:

```python
if self.mixing_intensity is None:
    mixing_power = self.kW_per_m3 * self.V_max
else:
    self._mixed.mix_from(self.ins)
    mixing_power = (
        self._mixed.mu * self.mixing_intensity**2 / 1000 * self.V_max
    )
```

Add a source comment identifying QSDsan static `Reactor.kW_per_m3`.

- [ ] **Step 2: Implement aeration power**

Prefer explicit `Q_air`; otherwise accept only `DiffusedAeration.Q_air`.
Convert `m3/d` to `m3/min`, then call `get_P_blower` with the configured blower
parameters. Add a source comment identifying QSDsan `get_P_blower`.

- [ ] **Step 3: Assign total utility**

Record `Aeration power`, `Mechanical mixing power`, and `Total power` in
`design_results`; set:

```python
self.power_utility.rate = aeration_power + mixing_power
```

- [ ] **Step 4: Run all focused tests**

Run:

```bash
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q
```

Expected: all tests pass.

### Task 5: Verify module compatibility

**Files:**
- Modify only if needed: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Run syntax and import checks**

```bash
/opt/anaconda3/envs/algae/bin/python -m compileall -q \
  exposan/pm2_ecorecover_lca/_sanunits.py
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -c \
  "from exposan.pm2_ecorecover_lca._sanunits import Tank; print(Tank.__name__)"
```

Expected: exit code 0 and output `Tank`.

- [ ] **Step 2: Run adjacent PM2 tests**

```bash
MPLCONFIGDIR=/tmp/algae-mpl /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py tests/test_pm2.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Review the final diff**

Confirm that:

- `ecorecover_lca.py` is unchanged;
- existing user edits in `_sanunits.py` remain;
- comments cite each requested source;
- no concrete construction cost is added.
