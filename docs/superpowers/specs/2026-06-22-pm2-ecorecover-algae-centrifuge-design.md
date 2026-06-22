# PM2 EcoRecover AlgaeCentrifuge SanUnit Design

## Goal

Add an `AlgaeCentrifuge` SanUnit to `exposan/pm2_ecorecover_lca/_sanunits.py`.
The unit will replace the plain `CENT = su.Splitter(...)` pattern when design,
purchase cost, stainless-steel mass, and centrifuge energy are needed. It will
inherit from `qsdsan.unit_operations.Splitter` and keep the inherited `_run()`
behavior unchanged.

## Class Interface

`AlgaeCentrifuge` will accept normal `Splitter` arguments plus:

- `algal_cell_diameter=5e-6`: algal particle diameter in `m`.
- `algal_particle_density=1050`: algal particle density in `kg/m3`.
- `water_density=1000`: water density in `kg/m3`.
- `T=25`: water temperature in `deg C` for viscosity.
- `phi=0`: solids volume fraction, dimensionless.
- `vgm=0.1e-6`: master-curve settling velocity in `m/s`.

`phi` will be used as a hindered-settling correction:

```python
vg_eff = vg * (1 - phi)**4.65
```

## Design Calculations

The design function will report influent flow in `m3/hr`, viscosity, gravity
settling velocity, effective settling velocity, master-curve flow, energy
intensity, and centrifuge stainless-steel weight.

Water viscosity will use the same relationship as `Ultrafiltration`:

```python
mu = 497e-3 / (T + 42.5)**1.5
```

Eq. 7 from Najjar and Abu-Shamleh (2020) will estimate terminal settling
velocity:

```python
vg = (rho_p - rho_w) * d**2 * g / (18 * mu)
```

Eq. 15 will convert the actual flow to the master-curve flow:

```python
Qm = Q_actual_m3_s * 3600 * vgm / vg_eff
```

Eq. 16 for disc centrifuges will estimate energy intensity:

```python
E_disc = 1.447 * Qm**(-0.304)
```

where `E_disc` is in `kWh/m3`. Power demand will be:

```python
power = E_disc * Q_m3_hr
```

The centrifuge weight will follow the Dolphin Centrifuge capacity correlation:

```python
weight = 1126.1 * np.log(Q_m3_hr) - 1204.8
```

where `Q_m3_hr` is influent flow in `m3/hr` and weight is in `kg`. If the flow
is zero, weight, power, and purchase cost will be zero.

## Cost Calculations

The centrifuge purchase cost will follow the BioSTEAM `LiquidsCentrifuge`
baseline algorithm:

```python
cost = 28100 * Q_m3_hr**0.574
```

The cost basis will use CEPCI `525.4`, upper-bound flow `100 m3/hr`, and
bare-module factor `2.03`, matching BioSTEAM's `LiquidsCentrifuge` pattern.
When flow exceeds `100 m3/hr`, the unit will use BioSTEAM-style parallel
centrifuge sizing:

```python
N = ceil(Q_m3_hr / 100)
Q_each = Q_m3_hr / N
cost_total = N * 28100 * Q_each**0.574
```

The energy intensity from Eq. 16 will replace BioSTEAM's default
`1.4 kW/(m3/hr)` centrifuge utility factor. The unit will set:

```python
power_utility.rate = E_disc * Q_m3_hr
```

## Validation

Validation will reject nonphysical inputs:

- `algal_cell_diameter > 0`
- `algal_particle_density > water_density`
- `water_density > 0`
- `T > -42.5`
- `0 <= phi < 1`
- `vgm > 0`

For positive flow, `vg_eff` and `Qm` must be positive. The unit will allow zero
flow and return zero cost, weight, and power.

## Testing

Add focused tests for:

- inherited `Splitter` mass-routing behavior;
- Eq. 7, hindered-settling correction, Eq. 15, Eq. 16, and power calculation;
- centrifuge purchase cost and parallel centrifuge count;
- stainless-steel weight correlation;
- zero-flow behavior;
- validation errors for invalid inputs.

## References

- Centrifuge cost correlation: BioSTEAM `LiquidsCentrifuge`.
- Centrifuge energy equations: Najjar and Abu-Shamleh, Algal Research 51
  (2020) 102046, http://doi.org/10.1016/j.algal.2020.102046.
- Centrifuge weight correlation data source:
  https://dolphincentrifuge.com/disc-stack-centrifuge/#sizing-and-capacity.
