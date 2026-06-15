# PM2 EcoRecover CO2 Supply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and export a pass-through `CO2Supply(SanUnit)` that estimates concentration-based CO2 makeup and its 2022-basis hourly operating cost without changing the process stream.

**Architecture:** Keep the class local to `exposan.pm2_ecorecover_lca._sanunits`. `_run()` copies the single inlet to the single outlet, `_design()` calculates CO2 demand from inlet concentration and flow, and `_cost()` records purchased CO2 as `add_OPEX`. Do not modify `ecorecover_lca.py`, PM2 kinetics, or stream mass balances.

**Tech Stack:** Python, QSDsan `SanUnit` and `WasteStream`, BioSTEAM CEPCI data, pytest.

---

### Task 1: Add focused CO2Supply behavior tests

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Read: `exposan/pm2_ecorecover_lca/_sanunits.py`

- [ ] **Step 1: Import CO2Supply**

Change the EXPOsan import to:

```python
from exposan.pm2_ecorecover_lca._sanunits import CO2Supply, Tank
```

- [ ] **Step 2: Add a CO2Supply fixture**

```python
@pytest.fixture
def co2_supply():
    pc.create_pm2_cmps()
    feed = WasteStream('co2_supply_feed')
    feed.set_flow_by_concentration(
        1000,
        {'S_CO2': 10},
        units=('L/hr', 'mg/L'),
    )
    return CO2Supply(
        'CO2_test',
        ins=feed,
        outs='co2_supply_effluent',
    )
```

- [ ] **Step 3: Test pass-through behavior and default demand**

```python
def test_co2_supply_passes_stream_and_calculates_default_makeup(co2_supply):
    feed = co2_supply.ins[0]
    co2_supply.simulate()
    effluent = co2_supply.outs[0]
    expected_base = (30 - 10) * 1000 * 1e-6
    expected_supply = expected_base * 1.10

    assert effluent.F_mass == pytest.approx(feed.F_mass)
    assert effluent.F_vol == pytest.approx(feed.F_vol)
    assert effluent.imass['S_CO2'] == pytest.approx(feed.imass['S_CO2'])
    assert co2_supply.design_results['Influent CO2 concentration'] == pytest.approx(10)
    assert co2_supply.design_results['Target CO2 concentration'] == pytest.approx(30)
    assert co2_supply.design_results['Wastewater flow'] == pytest.approx(1000)
    assert co2_supply.design_results['Base CO2 makeup'] == pytest.approx(expected_base)
    assert co2_supply.design_results['CO2 supply'] == pytest.approx(expected_supply)
    assert co2_supply.design_results['Excess CO2 fraction'] == pytest.approx(0.10)
```

- [ ] **Step 4: Test user-settable excess and no negative makeup**

```python
def test_co2_supply_uses_user_excess_fraction(co2_supply):
    co2_supply.excess_fraction = 0.25
    co2_supply.simulate()
    assert co2_supply.design_results['CO2 supply'] == pytest.approx(
        (30 - 10) * 1000 * 1e-6 * 1.25
    )


def test_co2_supply_has_zero_makeup_above_target(co2_supply):
    co2_supply.ins[0].set_flow_by_concentration(
        1000,
        {'S_CO2': 35},
        units=('L/hr', 'mg/L'),
    )
    co2_supply.simulate()
    assert co2_supply.design_results['Base CO2 makeup'] == 0
    assert co2_supply.design_results['CO2 supply'] == 0
```

- [ ] **Step 5: Test CEPCI-adjusted operating cost**

```python
def test_co2_supply_cost_uses_2022_cepci(co2_supply):
    co2_supply.simulate()
    supply = co2_supply.design_results['CO2 supply']
    expected_price = (
        45 / 1000
        * CEPCI_by_year[2022] / CEPCI_by_year[2016]
    )
    assert co2_supply.add_OPEX['CO2 supply'] == pytest.approx(
        supply * expected_price
    )
    assert not co2_supply.baseline_purchase_costs
```

Add this test import:

```python
from biosteam.units.design_tools import CEPCI_by_year
```

- [ ] **Step 6: Test zero flow and validation**

```python
def test_co2_supply_handles_zero_flow():
    pc.create_pm2_cmps()
    feed = WasteStream('empty_co2_supply_feed')
    unit = CO2Supply(
        'CO2_empty',
        ins=feed,
        outs='empty_co2_supply_effluent',
    )
    unit.simulate()
    assert unit.design_results['Wastewater flow'] == 0
    assert unit.design_results['CO2 supply'] == 0
    assert unit.add_OPEX['CO2 supply'] == 0


@pytest.mark.parametrize(
    ('name', 'value'),
    (
        ('target_CO2', -1),
        ('excess_fraction', -0.1),
        ('CO2_price', -1),
    ),
)
def test_co2_supply_rejects_negative_inputs(co2_supply, name, value):
    with pytest.raises(ValueError, match=name):
        setattr(co2_supply, name, value)


def test_co2_supply_rejects_invalid_component():
    pc.create_pm2_cmps()
    feed = WasteStream('invalid_co2_supply_feed', H2O=1, units='kg/hr')
    with pytest.raises(ValueError, match='CO2_ID'):
        CO2Supply(
            'CO2_invalid',
            ins=feed,
            outs='invalid_co2_supply_effluent',
            CO2_ID='missing_CO2',
        )
```

- [ ] **Step 7: Run the CO2Supply tests and verify RED**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py -k co2_supply -q
```

Expected: collection fails because `CO2Supply` is not defined or exported.

### Task 2: Implement the pass-through class and validation

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Export the class**

Add `'CO2Supply'` to `__all__` without changing the existing `Tank`,
`Photobioreactor`, or `Ecorecoverypump` exports:

```python
__all__ = (
    'Photobioreactor',
    'Ecorecoverypump',
    'Tank',
    'CO2Supply',
)
```

- [ ] **Step 2: Add the class interface**

Add after `Tank`:

```python
class CO2Supply(SanUnit):
    """Pass-through unit that estimates purchased CO2 makeup and cost."""

    _N_ins = 1
    _N_outs = 1
    _units = {
        'Influent CO2 concentration': 'mg/L',
        'Target CO2 concentration': 'mg/L',
        'Wastewater flow': 'L/hr',
        'Base CO2 makeup': 'kg/hr',
        'CO2 supply': 'kg/hr',
        'Excess CO2 fraction': '',
    }

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None,
            init_with='WasteStream', target_CO2=30.,
            excess_fraction=0.10, CO2_ID='S_CO2',
            CO2_price=45.,
        ):
        SanUnit.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo,
            init_with=init_with,
        )
        self.target_CO2 = target_CO2
        self.excess_fraction = excess_fraction
        self.CO2_ID = CO2_ID
        self.CO2_price = CO2_price
```

- [ ] **Step 3: Add nonnegative validated properties**

```python
@staticmethod
def _require_nonnegative(name, value):
    if value < 0:
        raise ValueError(f'`{name}` must be non-negative.')
    return value

@property
def target_CO2(self):
    return self._target_CO2

@target_CO2.setter
def target_CO2(self, value):
    self._target_CO2 = self._require_nonnegative('target_CO2', value)

@property
def excess_fraction(self):
    return self._excess_fraction

@excess_fraction.setter
def excess_fraction(self, value):
    self._excess_fraction = self._require_nonnegative(
        'excess_fraction', value,
    )

@property
def CO2_price(self):
    return self._CO2_price

@CO2_price.setter
def CO2_price(self, value):
    self._CO2_price = self._require_nonnegative('CO2_price', value)
```

- [ ] **Step 4: Validate the CO2 component ID**

```python
@property
def CO2_ID(self):
    return self._CO2_ID

@CO2_ID.setter
def CO2_ID(self, value):
    if value not in self.components.IDs:
        raise ValueError(
            f'`CO2_ID` must be one of the unit components; received {value!r}.'
        )
    self._CO2_ID = value
```

- [ ] **Step 5: Implement pass-through `_run()`**

```python
def _run(self):
    # Source: QSDsan Copier; this unit does not alter the mass balance.
    self.outs[0].copy_like(self.ins[0])
```

- [ ] **Step 6: Run interface tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py -k \
  "co2_supply_passes_stream or co2_supply_rejects" -q
```

Expected: pass-through and validation tests pass; design and cost assertions
remain failing because `_design()` and `_cost()` are not implemented.

### Task 3: Implement demand design and operating cost

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Implement `_design()`**

```python
def _design(self):
    D = self.design_results
    influent = self.ins[0]
    Q = influent.get_total_flow('L/hr')
    if Q > 0:
        C_in = float(
            influent.get_mass_concentration(
                'mg/L', IDs=(self.CO2_ID,),
            )[0]
        )
    else:
        C_in = 0.

    # mg/L * L/hr * 1e-6 = kg/hr.
    base_makeup = max(self.target_CO2 - C_in, 0.) * Q * 1e-6
    # User-settable allowance for unmodeled CO2 losses.
    supply = base_makeup * (1 + self.excess_fraction)

    D['Influent CO2 concentration'] = C_in
    D['Target CO2 concentration'] = self.target_CO2
    D['Wastewater flow'] = Q
    D['Base CO2 makeup'] = base_makeup
    D['CO2 supply'] = supply
    D['Excess CO2 fraction'] = self.excess_fraction
```

- [ ] **Step 2: Implement `_cost()`**

```python
def _cost(self):
    # Convert 2016 USD/metric tonne to 2022 USD/kg.
    price_2022 = (
        self.CO2_price / 1000
        * CEPCI_by_year[2022] / CEPCI_by_year[2016]
    )
    self.add_OPEX['CO2 supply'] = (
        self.design_results['CO2 supply'] * price_2022
    )
```

Do not add a baseline purchase cost, power utility, stream mass, construction,
or LCA inventory.

- [ ] **Step 3: Run all focused CO2Supply tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py -k co2_supply -q
```

Expected: all CO2Supply tests pass.

### Task 4: Verify integration and scope

**Files:**
- Modify only if verification identifies an issue:
  `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Run the full Tank and CO2Supply test module**

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent regression tests**

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m pytest \
  tests/test_pm2_ecorecover_lca.py tests/test_pm2.py \
  tests/test_module_conventions.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and import checks**

```bash
PYTHONPYCACHEPREFIX=/tmp/algae-pycache \
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -m compileall -q \
  exposan/pm2_ecorecover_lca/_sanunits.py

NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl \
  /opt/anaconda3/envs/algae/bin/python -c \
  "from exposan.pm2_ecorecover_lca._sanunits import CO2Supply; print(CO2Supply.__name__)"
```

Expected: exit code 0 and output `CO2Supply`.

- [ ] **Step 4: Review the final diff**

Confirm:

- `ecorecover_lca.py` is unchanged;
- `CO2Supply` is the only new production class;
- `_run()` does not alter the process stream;
- no CO2 mass is added to the outlet;
- no gas diffusivity or stoichiometric-demand model is introduced;
- no baseline purchase cost, construction, or electricity is added;
- the existing uncommitted `Tank` implementation remains intact.

- [ ] **Step 5: Commit only when explicitly requested**

The target production and test files already contain uncommitted Tank work.
Do not create an implementation commit unless the user explicitly asks to
commit the combined changes.
