# PM2 EcoRecover Tank Design

## Goal

Add a `Tank` sanitation unit to `exposan.pm2_ecorecover_lca._sanunits` that
retains the dynamic behavior and initialization interface of QSDsan's `CSTR`
while adding tank geometry, BioSTEAM `MixTank` purchase cost, and configurable
aeration and mechanical-mixing electricity. When aeration power is enabled, the
tank will also include screw-compressor and diffuser design and purchase cost.

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
- `diffuser_unit_cost`: user-supplied diffuser-grid cost in `USD/m2`, defaulting
  to zero as an explicit placeholder;
- `compressor_specific_mass`: user-supplied stainless-steel compressor mass in
  `kg/kW`, defaulting to zero as an explicit placeholder;
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

### Aeration equipment

Compressor and diffuser design is included only when
`include_aeration_power=True`.

The resolved airflow is converted from `m3/d` to actual cubic feet per minute.
The system-total number of parallel screw compressors is:

`ceil(Q_air_acfm / 20,000)`

following the maximum ACFM bound of BioSTEAM's screw-compressor cost algorithm.
Aeration power from `_get_aeration_power()` is treated as total compressor
driver power. Compressor stainless-steel mass is:

`aeration_power * compressor_specific_mass`

The default `compressor_specific_mass=0 kg/kW` is a documented placeholder
because no defensible general power-to-mass correlation was identified.

The diffuser grid is assumed to cover the full horizontal tank area:

`diffuser_area = W_tank * tank_length`

Its stainless-steel mass is:

`diffuser_area * 181 / 18.5`

where `181 / 18.5` has units of `kg/m2`, as specified for this model.

## Cost

The `_cost()` method will follow BioSTEAM's `MixTank` cost algorithm:

1. Retrieve the selected conventional mixing-tank purchase-cost algorithm.
2. Use `compute_number_of_tanks_and_purchase_cost` for the correlation volume.
3. Store the number of parallel tanks.
4. Store the baseline tank purchase cost before the material factor.
5. Apply the selected vessel material through `F_M`.

BioSTEAM's built-in fixed mixing-power assignment will not be used, because
power is calculated independently as described below.

When `include_aeration_power=True`, compressor purchase cost follows
BioSTEAM's `IsothermalCompressor` screw-compressor algorithm with an electric
motor:

`exp(8.2496 + 0.7243 * log(Pc))`

where `Pc` is driver power per compressor in hp. Total aeration power from
`_get_aeration_power()` is divided among the parallel compressors before
applying the correlation, and individual costs are summed. The cost is adjusted
from correlation CEPCI 567 to the current BioSTEAM CEPCI. A stainless-steel
material factor of 2.5 and bare-module factor of 2.15 are assigned to the
compressor cost entry.

The calculated compressor and diffuser costs are system totals. Because
BioSTEAM later multiplies all purchase-cost entries by `parallel['self']`, the
values stored in `baseline_purchase_costs` are divided by the number of
parallel tanks. This preserves the intended system-total aeration-equipment
cost.

If resolved airflow or aeration power is zero, compressor count and cost are
zero and the logarithmic cost correlation is not evaluated.

Because `_get_aeration_power()` already returns electrical power after blower
efficiency, no additional electric-motor efficiency adjustment is applied
during cost sizing.

Diffuser purchase cost is:

`diffuser_area * diffuser_unit_cost`

The default `diffuser_unit_cost=0 USD/m2` is a documented placeholder. Users
can supply a project-specific value without changing the cost method.

When aeration power is disabled, compressor and diffuser costs are omitted and
their aeration-specific design quantities are zero. No QSDsan `Construction`
objects are created for either item.

## Electricity

Aeration and mixing power are independently selectable and additive.

### Aeration

When `include_aeration_power` is true, airflow is selected in this order:

1. explicit `Q_air`, if provided;
2. `self.aeration.Q_air`, when the aeration object is `DiffusedAeration`;
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
- negative diffuser unit cost or compressor specific mass;
- unsupported vessel types or vessel materials.

## Source Comments

Short comments will identify the origin of each borrowed calculation:

- tank geometry: commented QSDsan dynamic `CSTR` design;
- purchase cost: BioSTEAM `MixTank`/`Tank`;
- aeration power: QSDsan `get_P_blower`;
- compressor sizing and purchase cost: BioSTEAM `Compressor`/
  `IsothermalCompressor` screw algorithm;
- diffuser area and mass: full-floor coverage assumption and the user-specified
  `181/18.5 kg/m2` factor;
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
  available;
- parallel compressor count from airflow;
- screw-compressor purchase cost and stainless-steel material factor;
- user-settable compressor specific mass;
- diffuser horizontal area, stainless-steel mass, and user-settable unit cost;
- omission of compressor and diffuser costs when aeration power is disabled.
