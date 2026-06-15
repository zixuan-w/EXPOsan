# PM2 EcoRecover Tank Design

## Goal

Add a `Tank` sanitation unit to `exposan.pm2_ecorecover_lca._sanunits` that
retains the dynamic behavior and initialization interface of QSDsan's `CSTR`
while adding tank geometry, BioSTEAM `MixTank` purchase cost, and configurable
aeration and mechanical-mixing electricity.

The existing `MIX = su.CSTR(...)` definition in `ecorecover_lca.py` will not be
changed.

## Class Interface

`Tank` will inherit from `qsdsan.unit_operations.CSTR` and accept all current
CSTR inputs used by the EcoRecover mixing reactor:

- streams, thermodynamics, split, and dynamic configuration;
- `V_max`, `W_tank`, `D_tank`, `freeboard`, `t_wall`, and `t_slab`;
- aeration, dissolved-oxygen, suspended-growth-model, gas-stripping, and
  exogenous-variable settings.

It will add:

- `V_wf`: working-volume fraction used by the BioSTEAM cost correlation;
- `vessel_type` and `vessel_material`: BioSTEAM `MixTank` cost selections;
- `include_aeration_power` and `include_mixing_power`;
- `Q_air`: optional explicit field airflow in `m3/d`;
- blower parameters accepted by `qsdsan.utils.get_P_blower`;
- `mixing_intensity`: optional velocity gradient `G` in `1/s`;
- `kW_per_m3`: specific mechanical-mixing power when `G` is absent.

If `mixing_intensity` is supplied, it takes precedence over `kW_per_m3`.

## Design

The `_design()` method will follow the commented QSDsan dynamic-CSTR design:

- tank volume is `V_max`;
- width is `W_tank`;
- liquid depth is `D_tank`;
- length is `V_max / (W_tank * D_tank)`;
- total wall depth includes freeboard;
- rectangular wall and slab concrete volumes are calculated from wall and slab
  thicknesses.

These concrete volumes are reported as design information only. They do not
create construction inventory or purchase cost.

The design results will also include the total volume passed to the BioSTEAM
cost algorithm. The correlation volume is `V_max / V_wf`, treating `V_max` as
working liquid volume.

## Cost

The `_cost()` method will follow BioSTEAM's `MixTank` cost algorithm:

1. Retrieve the selected conventional mixing-tank purchase-cost algorithm.
2. Use `compute_number_of_tanks_and_purchase_cost` for the correlation volume.
3. Store the number of parallel tanks.
4. Store the baseline tank purchase cost before the material factor.
5. Apply the selected vessel material through `F_M`.

BioSTEAM's built-in fixed mixing-power assignment will not be used, because
power is calculated independently as described below.

## Electricity

Aeration and mixing power are independently selectable and additive.

### Aeration

When `include_aeration_power` is true, airflow is selected in this order:

1. explicit `Q_air`, if provided;
2. `self.aeration.Q_air`, when the aeration object is `DiffusedAeration`.
3. otherwise, `0.1 * V_max * 1440` in `m3/d`, corresponding to an airflow
   rate of `0.1 m3/min` per `m3` of tank volume.

Airflow is converted from `m3/d` to `m3/min` and passed to
`qsdsan.utils.get_P_blower`.

### Mechanical mixing

When `include_mixing_power` is true:

- with no `mixing_intensity`, power is `kW_per_m3 * V_max`;
- with velocity gradient `G`, specific power is `mu * G**2 / 1000`, following
  QSDsan `Reactor`, and total power is that value times `V_max`.

The tank's `power_utility.rate` is the sum of enabled aeration and mixing
power. Separate design results record both terms.

## Validation

The class will reject:

- nonpositive tank width, liquid depth, wall thickness, slab thickness, and
  working-volume fraction;
- negative freeboard, airflow, mixing intensity, or specific mixing power;
- unsupported vessel types or vessel materials;

## Source Comments

Short comments will identify the origin of each borrowed calculation:

- tank geometry: commented QSDsan dynamic `CSTR` design;
- purchase cost: BioSTEAM `MixTank`/`Tank`;
- aeration power: QSDsan `get_P_blower`;
- mixing power: QSDsan static `Reactor`.

## Tests

Focused tests will verify:

- geometry and concrete-volume design results;
- BioSTEAM tank purchase-cost population;
- specific-power mechanical mixing;
- velocity-gradient mechanical mixing;
- explicit-airflow aeration power;
- `DiffusedAeration` airflow aeration power;
- combined and disabled power selections;
- default airflow based on tank volume when no other airflow source is
  available.
