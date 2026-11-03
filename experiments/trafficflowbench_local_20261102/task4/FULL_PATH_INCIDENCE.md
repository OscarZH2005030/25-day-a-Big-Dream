# Full static path incidence and OD margins

All 3,603 released path-link rows across ten networks satisfy the integer additive reconstruction from first-origin and last-destination anchors. The audit includes 2,072 link rows absent from the union of the three released Task4 count inputs. The joint linear rank adds zero beyond complete origin and destination margins on every network.

An independent graph traversal reconstructs integer origin and destination potentials for every link indicator, then checks all path edges. It confirms zero error throughout. This establishes the row-space result without a numerical rank threshold. The portable runner reproduces the tables and input hashes exactly.

Additional static rows can constrain unknown margins. Once the margins are fixed, any linear weighted sum of these released incidence rows also has zero additional OD matching information. This result concerns the released static path operator. It does not establish equivalence to actual detector measurements, ramp counts or time-resolved traffic observations.

```powershell
python task4/audit_full_path_incidence.py --release-root <release-root> --output <new-output-directory>
```
