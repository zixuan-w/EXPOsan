# PM2 EcoRecover Ultrafiltration SanUnit Design

## Goal

Add an `Ultrafiltration` SanUnit to `exposan/pm2_ecorecover_lca/_sanunits.py`.
The unit represents an ultrafiltration splitter with optional tank, air sparging,
and chemical-cleaning design and cost accounting. It should be separate from the
external `Membrane` helper class and should follow QSDsan/BioSTEAM unit patterns.

## Class Interface

`Ultrafiltration` will inherit from `qsdsan.unit_operations.Splitter`.
It will use the inherited `Splitter._run()` behavior and will not override
`_run()`.

The class will accept normal Splitter construction arguments plus:

- `R_t`: total membrane resistance in `1/m`.
- `T`: water temperature in deg C.
- `TMP`: transmembrane pressure in Pa.
- `capacity_factor=1.5`: multiplier on influent flow for design capacity.
- `include_tank=False`: whether to include tank volume and tank purchase cost.
- `include_sparging=False`: whether to include sparging airflow, compressor,
  diffusers, and blower power.
- `include_chemical_cleaning=False`: whether to include chemical usage and cost.
- `V_max=3.8`: tank volume in `m3`.
- `specific_sparging_air_demand=0.7`: sparging air demand in `m3/m2/hr`.
- `blower_efficiency=0.7`: compressor/blower efficiency, constrained to
  `(0, 1]`.
- `chemical_cleaning_frequency=1/45`: cleaning events per day.
- `citric_acid_concentration=2000`: cleaning concentration in `mg/L`.
- `sodium_hypochlorite_concentration=2000`: cleaning concentration in `mg/L`.
- `citric_acid_unit_price=1.06*euro_to_usd`: chemical price in `USD/kg`.
- `sodium_hypochlorite_unit_price=0.88/0.125*euro_to_usd`: chemical price in
  `USD/kg`.

To avoid adding module-count accounting now, membrane module area will be treated
as the total membrane area:

```python
A_mod = A
```

## Design Calculations

Membrane area will use the same viscosity, flux, and area relationships from
the external `Membrane` class:

```python
mu = 497e-3 / (T + 42.5)**1.5
J = TMP / mu / R_t
Q_design_m3_s = Q_in_m3_d * capacity_factor / 24 / 3600
A = Q_design_m3_s / J
```

If `include_tank=True`, the design results will include tank volume using
`V_max`. The tank design will intentionally not include concrete volume.

If `include_sparging=True`, sparging air flow will be:

```python
Q_air = specific_sparging_air_demand * A
```

where `Q_air` is in `m3/hr`. Blower power will use `qsdsan.utils.get_P_blower`
with `Q_air` converted to `m3/min`. Compressor count will use the BioSTEAM screw
compressor algorithm maximum ACFM bound, matching the method used in `Tank`.
Diffuser area will be the membrane area. Diffuser stainless steel will be:

```python
diffuser_stainless_steel = diffuser_area * 181 / 18.5
```

If `include_chemical_cleaning=True`, chemical usage will be based on tank volume,
cleaning frequency, and concentration:

```python
usage_kg_hr = V_max * 1000 * concentration_mg_L * 1e-6
              * chemical_cleaning_frequency / 24
```

Both citric acid and sodium hypochlorite are assumed to be used once per
cleaning event.

## Cost Calculations

Membrane purchase cost will use the external `Membrane` area-based cost equation:

```python
membrane_unit_cost = -2.985*np.log(A_mod) + 68.159
membrane_cost = membrane_unit_cost * euro_to_usd * A
```

If `include_tank=True`, tank purchase cost will use the same BioSTEAM MixTank
purchase-cost algorithm pattern already used in `Tank`.

If `include_sparging=True`, compressor purchase cost will use BioSTEAM's
`IsothermalCompressor.baseline_cost_algorithms['Screw']`, matching `Tank`.
Diffuser cost will be based on a user-settable `diffuser_unit_cost` in `USD/m2`
with a default value of zero when no project-specific diffuser price is
available.

Sparging blower power will be reported through `power_utility.rate`.

If `include_chemical_cleaning=True`, citric acid and sodium hypochlorite costs
will be added to `add_OPEX` as hourly costs:

```python
chemical_cost_USD_hr = usage_kg_hr * unit_price_USD_kg
```

## Validation

Validation will reject nonphysical inputs:

- positive values required for `R_t`, `TMP`, `capacity_factor`, and `V_max`;
- `T > -42.5` for the viscosity equation;
- `0 < blower_efficiency <= 1`;
- nonnegative values for concentrations, cleaning frequency, unit prices,
  specific air demand, diffuser unit cost, and compressor specific mass.

## Testing

Add focused tests for:

- inherited Splitter mass-routing behavior;
- membrane design area from `R_t`, `T`, `TMP`, influent flow, and
  `capacity_factor`;
- optional tank cost and design volume;
- optional sparging airflow, compressor count, diffuser mass, and power;
- optional chemical usage and OPEX;
- validation errors for invalid inputs.

## References

- Membrane area and membrane-cost equations: external `Membrane` helper in
  `/Users/zixuanwang/Dropbox/Ewing-Guest_Shared/membrane_61526.py`.
- Sparging demand, chemical cleaning frequency, and chemical price assumptions:
  https://doi.org/10.1016/j.scitotenv.2024.177273.
- Blower power: `qsdsan.utils.get_P_blower`, matching the method used in
  `Tank`.
- Compressor purchase cost: BioSTEAM
  `IsothermalCompressor.baseline_cost_algorithms['Screw']`, matching `Tank`.
