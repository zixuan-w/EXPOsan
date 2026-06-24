# PM2 EcoRecover Ultrafiltration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `Ultrafiltration` SanUnit that behaves like a `Splitter` and adds membrane, optional tank, optional sparging, and optional chemical-cleaning design and cost accounting.

**Architecture:** Implement one new class in `exposan/pm2_ecorecover_lca/_sanunits.py` and add focused tests to the existing PM2 EcoRecover LCA test module. The unit will inherit `su.Splitter`, leave `_run()` inherited, and keep all new accounting in `_design()` and `_cost()`.

**Tech Stack:** Python, QSDsan `WasteStream` and `su.Splitter`, BioSTEAM MixTank and screw-compressor cost algorithms, `pytest`.

---

## File Structure

- Modify `exposan/pm2_ecorecover_lca/_sanunits.py`
  - Export `Ultrafiltration`.
  - Add constants `euro_to_usd = 1.16` if not already present.
  - Add `Ultrafiltration` after `CO2Supply` or before `Photobioreactor`.
- Modify `tests/test_pm2_ecorecover_lca.py`
  - Import `Ultrafiltration`.
  - Add focused unit tests for splitter behavior, membrane design, optional sparging, optional tank cost, optional chemical cleaning, and input validation.

---

### Task 1: Add Failing Tests For Base Ultrafiltration Behavior

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add import and fixture**

Add NumPy to the top of the test file:

```python
import numpy as np
import pytest
```

Add `Ultrafiltration` to the existing import:

```python
from exposan.pm2_ecorecover_lca._sanunits import CO2Supply, Tank, Ultrafiltration
```

Add this fixture near the existing fixtures:

```python
@pytest.fixture
def ultrafiltration():
    pc.create_pm2_cmps()
    feed = WasteStream('uf_feed', H2O=1000, units='kg/hr')
    return Ultrafiltration(
        'UF_test',
        ins=feed,
        outs=('uf_permeate', 'uf_retentate'),
        split=0.4,
        R_t=1e14,
        T=25,
        TMP=2e5,
        capacity_factor=1.5,
    )
```

- [ ] **Step 2: Add inherited Splitter and membrane design tests**

Add these tests:

```python
def test_ultrafiltration_inherits_splitter_run(ultrafiltration):
    feed = ultrafiltration.ins[0]
    ultrafiltration.simulate()
    permeate, retentate = ultrafiltration.outs
    assert permeate.F_mass == pytest.approx(feed.F_mass * 0.4)
    assert retentate.F_mass == pytest.approx(feed.F_mass * 0.6)


def test_ultrafiltration_membrane_design(ultrafiltration):
    ultrafiltration.simulate()
    D = ultrafiltration.design_results
    Q = ultrafiltration.ins[0].get_total_flow('m3/d')
    mu = 497e-3 / (25 + 42.5)**1.5
    flux = 2e5 / mu / 1e14
    expected_area = Q * 1.5 / 24 / 3600 / flux

    assert D['Influent flow'] == pytest.approx(Q)
    assert D['Designed flow'] == pytest.approx(Q * 1.5)
    assert D['Water viscosity'] == pytest.approx(mu)
    assert D['Membrane flux'] == pytest.approx(flux)
    assert D['Membrane area'] == pytest.approx(expected_area)
    assert D['Membrane module area'] == pytest.approx(expected_area)
    assert ultrafiltration.baseline_purchase_costs['Membrane'] == pytest.approx(
        (-2.985 * np.log(expected_area) + 68.159) * 1.16 * expected_area
    )
    assert ultrafiltration.power_utility.rate == 0
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_inherits_splitter_run tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_membrane_design -q
```

Expected: fail with `ImportError` or `AttributeError` because `Ultrafiltration` is not defined yet.

---

### Task 2: Implement Core Ultrafiltration Class

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Export the class and add currency constant**

Update `__all__`:

```python
__all__ = ('Photobioreactor',
           'Ecorecoverypump',
           'Tank',
           'CO2Supply',
           'Ultrafiltration',
           )
```

Add near the conversion constants:

```python
euro_to_usd = 1.16
```

- [ ] **Step 2: Add the core class**

Insert this class after `CO2Supply`:

```python
class Ultrafiltration(su.Splitter):
    '''
    Splitter-based ultrafiltration unit with membrane design and optional
    support equipment accounting.
    '''
    _F_BM_default = {
        'Membrane': 1.,
        'Tank': 2.3,
        'Air compressor': 2.15,
        'Diffusers': 1.,
    }
    _units = {
        'Influent flow': 'm3/d',
        'Designed flow': 'm3/d',
        'Water viscosity': 'Pa*s',
        'Membrane flux': 'm3/m2/s',
        'Membrane area': 'm2',
        'Membrane module area': 'm2',
        'Tank volume': 'm3',
        'Sparging air flow': 'm3/hr',
        'Sparging air flow at compressor': 'cfm',
        'Number of air compressors': '',
        'Sparging power': 'kW',
        'Air compressor stainless steel': 'kg',
        'Diffuser area': 'm2',
        'Diffuser stainless steel': 'kg',
        'Citric acid usage': 'kg/hr',
        'Sodium hypochlorite usage': 'kg/hr',
    }
    purchase_cost_algorithms = mix_tank_purchase_cost_algorithms

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *, split,
            order=None, init_with='WasteStream', F_BM_default=None,
            isdynamic=False, R_t=1e14, T=25., TMP=2e5,
            capacity_factor=1.5, include_tank=False,
            include_sparging=False, include_chemical_cleaning=False,
            V_max=3.8, V_wf=0.8, vessel_type='Conventional',
            vessel_material='Stainless steel',
            specific_sparging_air_demand=0.7, blower_T=20,
            P_atm=101.325, P_inlet_loss=1, P_diffuser_loss=7,
            h_submergance=5.18, blower_efficiency=0.7,
            blower_K=0.283, diffuser_unit_cost=0.,
            compressor_specific_mass=0.,
            chemical_cleaning_frequency=1/45,
            citric_acid_concentration=2000.,
            sodium_hypochlorite_concentration=2000.,
            citric_acid_unit_price=1.06*euro_to_usd,
            sodium_hypochlorite_unit_price=0.88/0.125*euro_to_usd,
        ):
        su.Splitter.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        self.R_t = R_t
        self.T = T
        self.TMP = TMP
        self.capacity_factor = capacity_factor
        self.include_tank = bool(include_tank)
        self.include_sparging = bool(include_sparging)
        self.include_chemical_cleaning = bool(include_chemical_cleaning)
        self.V_max = V_max
        self.V_wf = V_wf
        self.vessel_type = vessel_type
        self.vessel_material = vessel_material
        self.specific_sparging_air_demand = specific_sparging_air_demand
        self.blower_T = blower_T
        self.P_atm = P_atm
        self.P_inlet_loss = P_inlet_loss
        self.P_diffuser_loss = P_diffuser_loss
        self.h_submergance = h_submergance
        self.blower_efficiency = blower_efficiency
        self.blower_K = blower_K
        self.diffuser_unit_cost = diffuser_unit_cost
        self.compressor_specific_mass = compressor_specific_mass
        self.chemical_cleaning_frequency = chemical_cleaning_frequency
        self.citric_acid_concentration = citric_acid_concentration
        self.sodium_hypochlorite_concentration = sodium_hypochlorite_concentration
        self.citric_acid_unit_price = citric_acid_unit_price
        self.sodium_hypochlorite_unit_price = sodium_hypochlorite_unit_price
```

- [ ] **Step 3: Add validation and membrane helper methods**

Add methods inside `Ultrafiltration`:

```python
    @staticmethod
    def _require_positive(name, value):
        if value <= 0:
            raise ValueError(f'`{name}` must be positive.')
        return value

    @staticmethod
    def _require_nonnegative(name, value):
        if value < 0:
            raise ValueError(f'`{name}` must be non-negative.')
        return value

    @property
    def R_t(self):
        return self._R_t

    @R_t.setter
    def R_t(self, value):
        self._R_t = self._require_positive('R_t', value)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value <= -42.5:
            raise ValueError('`T` must be greater than -42.5 deg C.')
        self._T = value

    @property
    def TMP(self):
        return self._TMP

    @TMP.setter
    def TMP(self, value):
        self._TMP = self._require_positive('TMP', value)

    @property
    def capacity_factor(self):
        return self._capacity_factor

    @capacity_factor.setter
    def capacity_factor(self, value):
        self._capacity_factor = self._require_positive('capacity_factor', value)

    @property
    def V_max(self):
        return self._V_max

    @V_max.setter
    def V_max(self, value):
        self._V_max = self._require_positive('V_max', value)

    @property
    def V_wf(self):
        return self._V_wf

    @V_wf.setter
    def V_wf(self, value):
        value = self._require_positive('V_wf', value)
        if value > 1:
            raise ValueError('`V_wf` cannot exceed 1.')
        self._V_wf = value

    def _get_membrane_design(self):
        Q = self.ins[0].get_total_flow('m3/d')
        Q_design = Q * self.capacity_factor
        mu = 497e-3 / (self.T + 42.5)**1.5
        J = self.TMP / mu / self.R_t
        A = Q_design / 24 / 3600 / J if Q_design else 0.
        return Q, Q_design, mu, J, A
```

- [ ] **Step 4: Add base design and membrane cost**

Add:

```python
    def _design(self):
        D = self.design_results
        Q, Q_design, mu, J, A = self._get_membrane_design()

        # Source: Membrane helper in membrane_61526.py.
        D['Influent flow'] = Q
        D['Designed flow'] = Q_design
        D['Water viscosity'] = mu
        D['Membrane flux'] = J
        D['Membrane area'] = A
        D['Membrane module area'] = A
        D['Tank volume'] = self.V_max if self.include_tank else 0.
        D['Sparging air flow'] = 0.
        D['Sparging air flow at compressor'] = 0.
        D['Number of air compressors'] = 0
        D['Sparging power'] = 0.
        D['Air compressor stainless steel'] = 0.
        D['Diffuser area'] = 0.
        D['Diffuser stainless steel'] = 0.
        D['Citric acid usage'] = 0.
        D['Sodium hypochlorite usage'] = 0.

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        A = D['Membrane area']

        if A > 0:
            # Source: Membrane helper in membrane_61526.py.
            membrane_unit_cost = -2.985 * np.log(D['Membrane module area']) + 68.159
            C['Membrane'] = membrane_unit_cost * euro_to_usd * A
        else:
            C['Membrane'] = 0.

        self.power_utility.rate = 0.
```

- [ ] **Step 5: Run base tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_inherits_splitter_run tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_membrane_design -q
```

Expected: PASS.

---

### Task 3: Add Failing Tests For Optional Tank, Sparging, And Cleaning

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add optional-feature tests**

Add these tests:

```python
def test_ultrafiltration_includes_tank_cost_when_enabled(ultrafiltration):
    ultrafiltration.include_tank = True
    ultrafiltration.V_max = 3.8
    ultrafiltration.simulate()
    assert ultrafiltration.design_results['Tank volume'] == pytest.approx(3.8)
    assert ultrafiltration.baseline_purchase_costs['Tank'] > 0


def test_ultrafiltration_sparging_design_and_cost(ultrafiltration):
    ultrafiltration.include_sparging = True
    ultrafiltration.specific_sparging_air_demand = 0.7
    ultrafiltration.diffuser_unit_cost = 12
    ultrafiltration.compressor_specific_mass = 5
    ultrafiltration.simulate()

    D = ultrafiltration.design_results
    A = D['Membrane area']
    Q_air = 0.7 * A
    power = get_P_blower(Q_air / 60, efficiency=0.7)
    hp = auom('kW').convert(power, 'hp')
    algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
    expected_compressor_cost = (
        CEPCI_by_year[2022] / algorithm.CE * algorithm.cost(hp)
    )

    assert D['Sparging air flow'] == pytest.approx(Q_air)
    assert D['Sparging air flow at compressor'] == pytest.approx(
        auom('m3/hr').convert(Q_air, 'cfm')
    )
    assert D['Number of air compressors'] == 1
    assert D['Sparging power'] == pytest.approx(power)
    assert D['Air compressor stainless steel'] == pytest.approx(power * 5)
    assert D['Diffuser area'] == pytest.approx(A)
    assert D['Diffuser stainless steel'] == pytest.approx(A * 181 / 18.5)
    assert ultrafiltration.baseline_purchase_costs[
        'Air compressor'
    ] == pytest.approx(expected_compressor_cost)
    assert ultrafiltration.baseline_purchase_costs['Diffusers'] == pytest.approx(
        A * 12
    )
    assert ultrafiltration.power_utility.rate == pytest.approx(power)


def test_ultrafiltration_chemical_cleaning_usage_and_cost(ultrafiltration):
    ultrafiltration.include_chemical_cleaning = True
    ultrafiltration.V_max = 3.8
    ultrafiltration.chemical_cleaning_frequency = 1 / 45
    ultrafiltration.citric_acid_concentration = 2000
    ultrafiltration.sodium_hypochlorite_concentration = 2000
    ultrafiltration.simulate()

    expected = 3.8 * 1000 * 2000 * 1e-6 * (1 / 45) / 24
    assert ultrafiltration.design_results['Citric acid usage'] == pytest.approx(
        expected
    )
    assert ultrafiltration.design_results[
        'Sodium hypochlorite usage'
    ] == pytest.approx(expected)
    assert ultrafiltration.add_OPEX['Citric acid'] == pytest.approx(
        expected * 1.06 * 1.16
    )
    assert ultrafiltration.add_OPEX['Sodium hypochlorite'] == pytest.approx(
        expected * 0.88 / 0.125 * 1.16
    )
```

- [ ] **Step 2: Run optional-feature tests to verify failure**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_includes_tank_cost_when_enabled tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_sparging_design_and_cost tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_chemical_cleaning_usage_and_cost -q
```

Expected: fail because optional tank, sparging, and chemical accounting are not implemented yet.

---

### Task 4: Implement Optional Feature Accounting

**Files:**
- Modify: `exposan/pm2_ecorecover_lca/_sanunits.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add remaining properties**

Add properties for `vessel_type`, `vessel_material`, `specific_sparging_air_demand`, `blower_efficiency`, `diffuser_unit_cost`, `compressor_specific_mass`, `chemical_cleaning_frequency`, concentrations, and unit prices:

```python
    @property
    def vessel_type(self):
        return self._vessel_type

    @vessel_type.setter
    def vessel_type(self, value):
        try:
            algorithm = self.purchase_cost_algorithms[value]
        except KeyError:
            valid = ', '.join(self.purchase_cost_algorithms)
            raise ValueError(
                f'`vessel_type` must be one of: {valid}.'
            ) from None
        self._vessel_type = value
        self.purchase_cost_algorithm = algorithm

    @property
    def vessel_material(self):
        return self._vessel_material

    @vessel_material.setter
    def vessel_material(self, value):
        try:
            factor = vessel_material_factors[value]
        except KeyError:
            valid = ', '.join(vessel_material_factors)
            raise ValueError(
                f'No vessel material factor is available for {value!r}; '
                f'choose one of: {valid}.'
            ) from None
        self._vessel_material = value
        self.F_M['Tank'] = factor

    @property
    def specific_sparging_air_demand(self):
        return self._specific_sparging_air_demand

    @specific_sparging_air_demand.setter
    def specific_sparging_air_demand(self, value):
        self._specific_sparging_air_demand = self._require_nonnegative(
            'specific_sparging_air_demand', value,
        )

    @property
    def blower_efficiency(self):
        return self._blower_efficiency

    @blower_efficiency.setter
    def blower_efficiency(self, value):
        value = self._require_positive('blower_efficiency', value)
        if value > 1:
            raise ValueError('`blower_efficiency` cannot exceed 1.')
        self._blower_efficiency = value

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

    @property
    def chemical_cleaning_frequency(self):
        return self._chemical_cleaning_frequency

    @chemical_cleaning_frequency.setter
    def chemical_cleaning_frequency(self, value):
        self._chemical_cleaning_frequency = self._require_nonnegative(
            'chemical_cleaning_frequency', value,
        )

    @property
    def citric_acid_concentration(self):
        return self._citric_acid_concentration

    @citric_acid_concentration.setter
    def citric_acid_concentration(self, value):
        self._citric_acid_concentration = self._require_nonnegative(
            'citric_acid_concentration', value,
        )

    @property
    def sodium_hypochlorite_concentration(self):
        return self._sodium_hypochlorite_concentration

    @sodium_hypochlorite_concentration.setter
    def sodium_hypochlorite_concentration(self, value):
        self._sodium_hypochlorite_concentration = self._require_nonnegative(
            'sodium_hypochlorite_concentration', value,
        )

    @property
    def citric_acid_unit_price(self):
        return self._citric_acid_unit_price

    @citric_acid_unit_price.setter
    def citric_acid_unit_price(self, value):
        self._citric_acid_unit_price = self._require_nonnegative(
            'citric_acid_unit_price', value,
        )

    @property
    def sodium_hypochlorite_unit_price(self):
        return self._sodium_hypochlorite_unit_price

    @sodium_hypochlorite_unit_price.setter
    def sodium_hypochlorite_unit_price(self, value):
        self._sodium_hypochlorite_unit_price = self._require_nonnegative(
            'sodium_hypochlorite_unit_price', value,
        )
```

- [ ] **Step 2: Extend `_design()`**

Add these branches at the end of `_design()`:

```python
        if self.include_sparging:
            Q_air = self.specific_sparging_air_demand * A
            power = self._get_sparging_power(Q_air)
            algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
            Q_air_acfm = auom('m3/hr').convert(Q_air, 'cfm')
            max_acfm = algorithm.acfm_bounds[1]
            N_compressors = ceil(Q_air_acfm / max_acfm) if Q_air_acfm > 0 else 0
            # Source: diffuser stainless steel method used in Tank.
            D['Sparging air flow'] = Q_air
            D['Sparging air flow at compressor'] = Q_air_acfm
            D['Number of air compressors'] = N_compressors
            D['Sparging power'] = power
            D['Air compressor stainless steel'] = (
                power * self.compressor_specific_mass
            )
            D['Diffuser area'] = A
            D['Diffuser stainless steel'] = A * 181 / 18.5

        if self.include_chemical_cleaning:
            D['Citric acid usage'] = self._chemical_usage(
                self.citric_acid_concentration,
            )
            D['Sodium hypochlorite usage'] = self._chemical_usage(
                self.sodium_hypochlorite_concentration,
            )
```

Add helper methods:

```python
    def _get_sparging_power(self, Q_air):
        if not self.include_sparging:
            return 0.
        # Source: QSDsan wwt_design.get_P_blower; it expects m3/min.
        return get_P_blower(
            Q_air / 60,
            T=self.blower_T,
            P_atm=self.P_atm,
            P_inlet_loss=self.P_inlet_loss,
            P_diffuser_loss=self.P_diffuser_loss,
            h_submergance=self.h_submergance,
            efficiency=self.blower_efficiency,
            K=self.blower_K,
        )

    def _chemical_usage(self, concentration):
        return (
            self.V_max * 1000 * concentration * 1e-6
            * self.chemical_cleaning_frequency / 24
        )
```

- [ ] **Step 3: Extend `_cost()`**

Add after membrane cost:

```python
        if self.include_tank:
            N, Cp = compute_number_of_tanks_and_purchase_cost(
                self.V_max / self.V_wf, self.purchase_cost_algorithm,
            )
            default_material = self.purchase_cost_algorithm.material
            C['Tank'] = N * Cp / vessel_material_factors.get(default_material, 1.)
        else:
            C.pop('Tank', None)

        if self.include_sparging:
            N_compressors = D['Number of air compressors']
            sparging_power = D['Sparging power']
            if N_compressors and sparging_power > 0:
                algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
                total_hp = auom('kW').convert(sparging_power, 'hp')
                hp_per_compressor = total_hp / N_compressors
                compressor_cost = (
                    N_compressors * CEPCI_by_year[2022] / algorithm.CE
                    * algorithm.cost(hp_per_compressor)
                )
            else:
                compressor_cost = 0.
            C['Air compressor'] = compressor_cost
            C['Diffusers'] = D['Diffuser area'] * self.diffuser_unit_cost
            self.F_M['Air compressor'] = 2.5
        else:
            C.pop('Air compressor', None)
            C.pop('Diffusers', None)

        if self.include_chemical_cleaning:
            self.add_OPEX['Citric acid'] = (
                D['Citric acid usage'] * self.citric_acid_unit_price
            )
            self.add_OPEX['Sodium hypochlorite'] = (
                D['Sodium hypochlorite usage']
                * self.sodium_hypochlorite_unit_price
            )
        else:
            self.add_OPEX.pop('Citric acid', None)
            self.add_OPEX.pop('Sodium hypochlorite', None)

        self.power_utility.rate = D['Sparging power']
```

- [ ] **Step 4: Run optional-feature tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_includes_tank_cost_when_enabled tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_sparging_design_and_cost tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_chemical_cleaning_usage_and_cost -q
```

Expected: PASS.

---

### Task 5: Add Validation Tests And Final Verification

**Files:**
- Modify: `tests/test_pm2_ecorecover_lca.py`
- Test: `tests/test_pm2_ecorecover_lca.py`

- [ ] **Step 1: Add validation tests**

Add:

```python
@pytest.mark.parametrize(
    ('name', 'value'),
    (
        ('R_t', 0),
        ('TMP', 0),
        ('capacity_factor', 0),
        ('V_max', 0),
        ('specific_sparging_air_demand', -1),
        ('blower_efficiency', 0),
        ('blower_efficiency', 1.1),
        ('diffuser_unit_cost', -1),
        ('compressor_specific_mass', -1),
        ('chemical_cleaning_frequency', -1),
        ('citric_acid_concentration', -1),
        ('sodium_hypochlorite_concentration', -1),
        ('citric_acid_unit_price', -1),
        ('sodium_hypochlorite_unit_price', -1),
    ),
)
def test_ultrafiltration_rejects_invalid_inputs(ultrafiltration, name, value):
    with pytest.raises(ValueError, match=name):
        setattr(ultrafiltration, name, value)


def test_ultrafiltration_rejects_invalid_temperature(ultrafiltration):
    with pytest.raises(ValueError, match='T'):
        ultrafiltration.T = -42.5
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_rejects_invalid_inputs tests/test_pm2_ecorecover_lca.py::test_ultrafiltration_rejects_invalid_temperature -q
```

Expected: PASS.

- [ ] **Step 3: Run full PM2 EcoRecover LCA tests**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m pytest tests/test_pm2_ecorecover_lca.py -q
```

Expected: PASS.

- [ ] **Step 4: Compile the edited module**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/algae-mpl PYTHONPYCACHEPREFIX=/tmp/algae-pycache /opt/anaconda3/envs/algae/bin/python -m py_compile exposan/pm2_ecorecover_lca/_sanunits.py
```

Expected: command exits with code 0.

- [ ] **Step 5: Review changed files**

Run:

```bash
git status --short
git diff -- exposan/pm2_ecorecover_lca/_sanunits.py tests/test_pm2_ecorecover_lca.py
```

Expected: only `_sanunits.py` and `tests/test_pm2_ecorecover_lca.py` contain implementation/test changes for `Ultrafiltration`; the plan file may also be uncommitted.
