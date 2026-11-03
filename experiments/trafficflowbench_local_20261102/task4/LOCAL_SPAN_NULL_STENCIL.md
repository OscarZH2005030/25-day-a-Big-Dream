# A local OD ambiguity with twelve changing cells

Take the outer product of the second-difference vector `[1, -2, 1]` with itself. Subtract a copy shifted one row and one column on the OD grid. The resulting 4×4 stencil has twelve nonzero cells.

Each block has zero row and column sums and zero row and column first moments. The two blocks contribute identical span histograms because diagonal translation preserves destination minus origin. Their difference therefore preserves the global span histogram as well. The construction uses zone indices and assumes each affected path exists.

An enumeration on the frozen flow found feasible placements in all twenty validation/private network groups. For each group it used half the largest feasible step along one placement, preserving nonnegativity and the original-prior orthant and L1 distance. The relative OD L1 movements range from 0.000807 to 0.003675. These are single local witnesses, distinct from the larger LP witnesses.

Independent checks verified the integer identities, placement counts, step lengths and released count invariance. The portable runner reproduced all twenty results exactly. This is a structural ambiguity result under hypothetical summaries and does not supply an observation or a candidate.

```powershell
python task4/audit_local_span_stencil.py --release-root <release-root> --output <new-output-directory>
```
