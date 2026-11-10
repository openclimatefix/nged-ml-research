# Gross-demand simulation

**It is worth running one small known-answer test of a simulator that fits, to each Primary
substation's metered power, a list of the assets behind the meter and a few parameters per asset.**
The simulator's value would be an asset list per Primary and scenarios mostly at long horizons,
not a better short-range forecast. NGED cannot confirm the asset list beyond the register, so the
list is shown as inferred. Gross demand is the demand before the embedded generation behind a substation
is netted off; a Primary's meter records only the net. No experiment in this theme has run yet.

**The asset list has to come from NGED's Embedded Capacity Register.** In the four experiments read
for this theme that let a fit decide on its own whether a generator exists, the fit reported
generators that are absent or unconfirmed. At the 16 Primaries in the study area the register lists
98 connected generation or storage entries (one technology at one site each), most of them small
solar. The register omits assets below 50 kW, and in the one case checked, the register names a
different Primary from the one whose meter records a 6 MW gas engine.

## Pages

- [An asset-inventory simulator](asset-inventory-simulator.md): what fitting an asset list with a
  few parameters per asset, instead of the state of each asset at every half-hour, can and cannot
  identify; what the register and the repository already supply; the maintainer's answers; what a forward
  model with real operating logic means and which parts exist; the candidate uses ranked; the
  smallest test; the risks; and the open question for the maintainer.
- [The simulator discussion of 2026-11-09](simulator-discussion-2026-11-09.md): the take-home of the
  discussion session that started this theme, with the discussion's numbers checked against merged
  pages.
- [Experiments](experiments.md) and [Research plan](plan.md): empty until an experiment is agreed.
