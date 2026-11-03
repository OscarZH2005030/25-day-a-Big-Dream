# Which marginal quantities are identified by the released count operator

For each of the thirty network/input groups, the released count row space equals the space of net zone flows. A net zone flow is the origin total minus the destination total at the same zone. The count operator determines these linear quantities under the static exact-count interpretation.

|Query|Identified|Total tested|
|---|---:|---:|
|Net zone flow|2,445|2,445|
|Origin total|30|2,415|
|Destination total|30|2,415|
|Total demand|0|30|

The identified origin and destination totals are the first and last zones respectively, where only one side of the margin exists. The independent certificate reconstructs each count row as an exact difference of common zone potentials. Modular elimination certifies rank equal to the number of zones minus one. A triangle exchange replaces a first-to-last trip with one first-to-interior and one interior-to-last trip. This leaves net flows unchanged and alters the interior gross margins and total trips.

The portable audit reproduces all thirty groups exactly. The structural result does not establish that a hidden reference matches the released counts. It uses the released operator and does not introduce real detector data.

```powershell
python task4/audit_released_marginal_information.py --release-root <release-root> --output <new-output-directory>
```
