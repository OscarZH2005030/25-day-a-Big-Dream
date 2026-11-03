# Origin timing aliasing at unit-bin resolution

In the idealized model, a trip from origin `o` reaches cut `k` at `k + epoch[o] - o`. Origins with the same offset `epoch[o] - o` always share an arrival stream. Given origin margins, consecutive stream counts recover the destination totals for each stream. The joint rank is the sum of origin/destination graph ranks across streams.

|Known origin pulse epochs|Rank added beyond OD margins|Remaining linear nullity|
|---|---:|---:|
|Simultaneous|561–4,851|0|
|Integer jitter from -4 through 4|433–3,293|128–1,802|
|Half compensation|289–2,450|272–2,401|
|Full compensation|0|561–4,851|

Seed 2026110343 fixes the integer draw and jitter per network. The original audit checks exact count differencing and four-cell null directions. A separate graph traversal verifies all 40 ranks. The portable run matches every result table and input hash.

Unit resolution alone is insufficient when distinct origins produce the same arrivals. This experiment assumes that the hypothetical observer knows the epochs and delays. It does not infer any actual schedules, travel times or detector compatibility from Task4.

```powershell
python task4/audit_hypothetical_time_alias.py --release-root <release-root> --output <new-output-directory>
```
