# Two L1 repair objectives at identical generated counts

The comparison reuses forty artificial cases from the source-clipping stress. Two LPs match the same generated counts and minimize distance either to the supplied prior or to its signed minimum-L2 net repair. HiGHS chooses among tied solutions. The previous soft-fit prediction is a descriptive reference with a different count constraint.

|Artificial mechanism|Objective|Mean OD score|Mean Task4 composite|
|---|---|---:|---:|
|Shared lognormal|Soft reference|0.164477|0.577252|
|Shared lognormal|Prior L1|0.231096|0.585705|
|Shared lognormal|Invisible L1|0.214917|0.598981|
|Shared positive background|Soft reference|0.243031|0.600404|
|Shared positive background|Prior L1|0.110742|0.520360|
|Shared positive background|Invisible L1|0.225340|0.590405|

On these reused lognormal cases, the two exact-count objectives rank differently on OD and composite. In shared positive background both exact-count solutions trail the soft reference on average. This comparison depends on the artificial cases and solver tie-breaking. It does not validate hidden OD or a real candidate.

Independent readback checked all 120 action scores and exact-count constraints. The portable chain rebuilt the input cases, matched the four result files and reproduced forty artifact hashes. All flow vectors remain local.

```powershell
python task4/stress_l1_repair.py --release-root <release-root> --output <new-output-directory>
```
