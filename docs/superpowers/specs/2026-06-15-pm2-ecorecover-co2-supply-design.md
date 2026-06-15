# PM2 EcoRecover CO2 Supply Design

## Goal

Add a `CO2Supply` sanitation unit to
`exposan.pm2_ecorecover_lca._sanunits`. The unit will pass its influent to its
effluent unchanged while estimating the purchased CO2 required to raise the
influent `S_CO2` concentration to a target value.

Only the class and its export will be added. `ecorecover_lca.py` will not be
modified.

## Scope

`CO2Supply` will inherit from `qsdsan.SanUnit` and have one inlet and one
outlet. It is a design and operating-cost bookkeeping unit, not a physical CO2
mass-balance model.

The unit will not:

- add CO2 to the outlet stream;
- alter PM2 reaction kinetics or initial conditions;
- model gas-liquid diffusion, outgassing, or carbonate equilibrium;
- add construction inventory, equipment purchase cost, or electricity.

## Interface

The constructor will accept:

- `target_CO2`: target `S_CO2` concentration in `mg CO2/L`, default `30`;
- `excess_fraction`: fractional excess applied to concentration-based makeup,
  default `0.10`;
- `CO2_ID`: component ID used for dissolved CO2, default `S_CO2`;
- `CO2_price`: CO2 price in 2016 `USD/metric tonne`, default `45`;
- standard `SanUnit` stream and thermodynamic arguments.

The class will reject negative target concentration, excess fraction, or CO2
price. It will raise a clear error when `CO2_ID` is not present in the
thermodynamic component set.

## Stream Behavior

The `_run()` method will copy the inlet to the outlet unchanged:

```python
self.outs[0].copy_like(self.ins[0])
```

This preserves flow, composition, temperature, pressure, and phase. The
calculated supply rate remains a design result only.

## Design

The `_design()` method will read:

- influent `S_CO2` concentration using
  `get_mass_concentration('mg/L', IDs=(CO2_ID,))`;
- influent wastewater flow using `get_total_flow('L/hr')`.

The base concentration makeup is:

```text
base_CO2 = max(target_CO2 - influent_CO2, 0)
           * wastewater_flow * 1e-6
```

where concentration is in `mg/L`, flow is in `L/hr`, and `base_CO2` is in
`kg/hr`.

The final purchased supply rate is:

```text
CO2_supply = base_CO2 * (1 + excess_fraction)
```

The default 10% excess is a simple allowance for unmodeled CO2 losses. It is
not calculated from stoichiometric demand or gas diffusivity.

Design results will include:

- `Influent CO2 concentration`, `mg/L`;
- `Target CO2 concentration`, `mg/L`;
- `Wastewater flow`, `L/hr`;
- `Base CO2 makeup`, `kg/hr`;
- `CO2 supply`, `kg/hr`;
- `Excess CO2 fraction`, dimensionless.

An empty influent with zero volumetric flow will produce zero makeup and supply
rather than raising a concentration error.

## Operating Cost

The default CO2 price is `45 USD/metric tonne` in 2016 dollars. It will be
converted to the 2022 cost basis:

```text
CO2_price_2022 = 45 / 1000
                 * CEPCI_by_year[2022] / CEPCI_by_year[2016]
```

in `USD/kg`.

The `_cost()` method will calculate:

```text
CO2_supply_cost = CO2_supply * CO2_price_2022
```

in `USD/hr`, and store it as:

```python
self.add_OPEX['CO2 supply']
```

No baseline purchase cost will be created.

## Source Comments

Short comments in the class will identify:

- the pass-through stream behavior;
- the concentration-times-flow unit conversion;
- the 2016-to-2022 CEPCI price adjustment;
- the excess fraction as a user-settable empirical loss allowance.

## Tests

Focused tests will verify:

- inlet and outlet are identical after `_run()`;
- makeup from target minus influent concentration;
- zero makeup when influent concentration meets or exceeds the target;
- default and user-specified excess fractions;
- correct `mg/L * L/hr` to `kg/hr` conversion;
- correct 2016-to-2022 CO2 price adjustment and hourly `add_OPEX`;
- zero-flow behavior;
- validation of negative inputs and an invalid `CO2_ID`;
- export of `CO2Supply` from `_sanunits.py`;
- no modification to `ecorecover_lca.py`.
