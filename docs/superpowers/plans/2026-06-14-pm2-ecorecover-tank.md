# PM2 EcoRecover Tank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dynamic `Tank(CSTR)` sanitation unit with rectangular-tank design results, BioSTEAM `MixTank` equipment cost, independently selectable aeration and mechanical-mixing electricity, and aeration-equipment design and cost.

**Architecture:** Keep process dynamics entirely in the inherited QSDsan `CSTR`. Add only design, purchase-cost, validation, and utility accounting in the EXPOsan subclass. Reuse BioSTEAM's public tank and screw-compressor cost algorithms and QSDsan's public blower helper instead of duplicating their equations.

**Tech Stack:** Python, QSDsan dynamic `CSTR`, BioSTEAM tank and compressor cost algorithms, pytest.

---

### Amendment: Add air compressor and diffuser equipment

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`

- [ ] **Step 1: Add imports used to calculate expected compressor results**

Add these imports to `tests/test_pm2_ecorecover_lca.py`:

```python
import biosteam as bst

from biosteam.units.compressor import IsothermalCompressor
from qsdsan.utils import auom, get_P_blower
```

Keep the existing QSDsan and EXPOsan imports.

- [ ] **Step 2: Add a failing enabled-equipment design and cost test**

```python
def test_tank_adds_aeration_equipment_when_enabled(tank):
    tank.include_aeration_power = True
    tank.Q_air = 1440
    tank.diffuser_unit_cost = 12
    tank.compressor_specific_mass = 5
    tank.simulate()

    D = tank.design_results
    N_tanks = tank.parallel['self']
    power = get_P_blower(1)
    hp = auom('kW').convert(power, 'hp')
    algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
    expected_compressor_cost = bst.CE / algorithm.CE * algorithm.cost(hp)
    expected_area = tank.W_tank * D['Tank length']

    assert D['Air flow rate'] == pytest.approx(1440)
    assert D['Air flow rate at compressor'] == pytest.approx(
        auom('m3/d').convert(1440, 'cfm')
    )
    assert D['Number of air compressors'] == 1
    assert D['Air compressor stainless steel'] == pytest.approx(power * 5)
    assert D['Diffuser area'] == pytest.approx(expected_area)
    assert D['Diffuser stainless steel'] == pytest.approx(
        expected_area * 181 / 18.5
    )
    assert (
        tank.baseline_purchase_costs['Air compressor'] * N_tanks
        == pytest.approx(expected_compressor_cost)
    )
    assert (
        tank.baseline_purchase_costs['Diffusers'] * N_tanks
        == pytest.approx(expected_area * 12)
    )
    assert tank.F_M['Air compressor'] == pytest.approx(2.5)
    assert tank.F_BM['Air compressor'] == pytest.approx(2.15)
```

- [ ] **Step 3: Add failing parallel, disabled, and validation tests**

```python
def test_tank_sizes_parallel_air_compressors(tank):
    tank.include_aeration_power = True
    tank.Q_air = auom('cfm').convert(40001, 'm3/d')
    tank.simulate()
    assert tank.design_results['Number of air compressors'] == 3


def test_tank_omits_aeration_equipment_when_disabled(tank):
    tank.diffuser_unit_cost = 12
    tank.compressor_specific_mass = 5
    tank.simulate()
    D = tank.design_results
    assert D['Number of air compressors'] == 0
    assert D['Air compressor stainless steel'] == 0
    assert D['Diffuser area'] == 0
    assert D['Diffuser stainless steel'] == 0
    assert 'Air compressor' not in tank.baseline_purchase_costs
    assert 'Diffusers' not in tank.baseline_purchase_costs


def test_tank_handles_zero_airflow_without_compressor_cost(tank):
    tank.include_aeration_power = True
    tank.Q_air = 0
    tank.simulate()
    assert tank.design_results['Number of air compressors'] == 0
    assert tank.baseline_purchase_costs['Air compressor'] == 0


@pytest.mark.parametrize(
    'name',
    ('diffuser_unit_cost', 'compressor_specific_mass'),
)
def test_tank_rejects_negative_aeration_equipment_factors(tank, name):
    with pytest.raises(ValueError, match=name):
        setattr(tank, name, -1)
```

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py::test_tank_adds_aeration_equipment_when_enabled \
  tests/test_pm2_ecorecover_lca.py::test_tank_sizes_parallel_air_compressors \
  tests/test_pm2_ecorecover_lca.py::test_tank_omits_aeration_equipment_when_disabled \
  tests/test_pm2_ecorecover_lca.py::test_tank_handles_zero_airflow_without_compressor_cost \
  tests/test_pm2_ecorecover_lca.py::test_tank_rejects_negative_aeration_equipment_factors \
  -q
```

Expected: FAIL because the constructor parameters, design-result keys, and
aeration-equipment costs do not exist yet.

- [ ] **Step 5: Add BioSTEAM compressor imports and design-result units**

In `exposan/pm2_ecorecover_lca/_sanunits.py`, add:

```python
import biosteam as bst

from biosteam.units.compressor import IsothermalCompressor
```

Extend `Tank._units` with:

```python
'Air flow rate': 'm3/d',
'Air flow rate at compressor': 'cfm',
'Number of air compressors': '',
'Air compressor stainless steel': 'kg',
'Diffuser area': 'm2',
'Diffuser stainless steel': 'kg',
```

Extend the class-level bare-module defaults:

```python
_F_BM_default = {
    'Tank': 2.3,
    'Air compressor': 2.15,
    'Diffusers': 1.0,
}
```

- [ ] **Step 6: Add and validate the user-settable placeholder factors**

Add constructor parameters:

```python
diffuser_unit_cost=0.,
compressor_specific_mass=0.,
```

Assign them through validated properties:

```python
@property
def diffuser_unit_cost(self):
    return self._diffuser_unit_cost

@diffuser_unit_cost.setter
def diffuser_unit_cost(self, value):
    self._diffuser_unit_cost = self._require_nonnegative(
        'diffuser_unit_cost', value,
    )

@property
def compressor_specific_mass(self):
    return self._compressor_specific_mass

@compressor_specific_mass.setter
def compressor_specific_mass(self, value):
    self._compressor_specific_mass = self._require_nonnegative(
        'compressor_specific_mass', value,
    )
```

Document both zero defaults as placeholders in the class docstring.

- [ ] **Step 7: Centralize airflow resolution**

Add:

```python
def _get_Q_air(self):
    Q_air = self.Q_air
    if Q_air is None and isinstance(self.aeration, pc.DiffusedAeration):
        Q_air = self.aeration.Q_air
    if Q_air is None:
        Q_air = 0.1 * self.V_max * 1440
    return Q_air
```

Change `_get_aeration_power()` to call `_get_Q_air()` and retain the existing
`m3/d` to `m3/min` conversion passed to `get_P_blower`.

- [ ] **Step 8: Add compressor and diffuser design**

At the end of `_design()`, add:

```python
if self.include_aeration_power:
    Q_air = self._get_Q_air()
    Q_air_acfm = auom('m3/d').convert(Q_air, 'cfm')
    aeration_power = self._get_aeration_power()
    compressor_algorithm = (
        IsothermalCompressor.baseline_cost_algorithms['Screw']
    )
    max_acfm = compressor_algorithm.acfm_bounds[1]
    N_compressors = ceil(Q_air_acfm / max_acfm) if Q_air_acfm > 0 else 0
    diffuser_area = W * L
else:
    Q_air = Q_air_acfm = aeration_power = 0.
    N_compressors = 0
    diffuser_area = 0.

D['Air flow rate'] = Q_air
D['Air flow rate at compressor'] = Q_air_acfm
D['Number of air compressors'] = N_compressors
D['Air compressor stainless steel'] = (
    aeration_power * self.compressor_specific_mass
)
D['Diffuser area'] = diffuser_area
D['Diffuser stainless steel'] = diffuser_area * 181 / 18.5
```

Add source comments for BioSTEAM's screw-compressor ACFM bound and the specified
full-floor diffuser assumption.

- [ ] **Step 9: Calculate power once, then add equipment purchase cost**

After the Tank cost and after `N = self.parallel['self']` is known, calculate
and record the power terms before aeration-equipment cost:

```python
aeration_power = self._get_aeration_power()
mixing_power = self._get_mixing_power()
total_power = aeration_power + mixing_power
D['Aeration power'] = aeration_power
D['Mechanical mixing power'] = mixing_power
D['Total power'] = total_power
```

Reuse the local `aeration_power` in the compressor cost:

```python
if self.include_aeration_power:
    N_compressors = D['Number of air compressors']
    if N_compressors and aeration_power > 0:
        algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
        total_hp = auom('kW').convert(aeration_power, 'hp')
        hp_per_compressor = total_hp / N_compressors
        compressor_cost = (
            N_compressors * bst.CE / algorithm.CE
            * algorithm.cost(hp_per_compressor)
        )
    else:
        compressor_cost = 0.

    C['Air compressor'] = compressor_cost / N
    C['Diffusers'] = (
        D['Diffuser area'] * self.diffuser_unit_cost / N
    )
    self.F_M['Air compressor'] = 2.5
else:
    C.pop('Air compressor', None)
    C.pop('Diffusers', None)

self.power_utility.rate = total_power / N
```

The division by `N` is required because BioSTEAM later multiplies every
purchase-cost entry by `parallel['self']`. Add a source comment identifying the
BioSTEAM `IsothermalCompressor` screw correlation. Do not apply another motor
efficiency because `_get_aeration_power()` already returns electrical power.
Remove the old duplicate power calculation from the end of `_cost()`.

- [ ] **Step 10: Run the focused aeration-equipment tests and verify GREEN**

Run the command from Step 4.

Expected: all selected tests pass.

- [ ] **Step 11: Run the complete Tank and adjacent regression tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py tests/test_pm2.py \
  tests/test_module_conventions.py -q
```

Expected: all tests pass.

- [ ] **Step 12: Review the final implementation diff**

Confirm:

- `ecorecover_lca.py` is unchanged;
- compressor and diffuser equipment appears only when
  `include_aeration_power=True`;
- existing Tank power is not added a second time;
- system-total compressor and diffuser cost is not multiplied twice by
  `parallel['self']`;
- `diffuser_unit_cost` and `compressor_specific_mass` remain explicit
  zero-default placeholders;
- no QSDsan `Construction` objects are added.

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
