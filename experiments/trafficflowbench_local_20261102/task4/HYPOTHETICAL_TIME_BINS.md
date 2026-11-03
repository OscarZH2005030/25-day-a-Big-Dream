# OD information in idealized time-resolved zone-cut counts

This experiment assumes one known simultaneous departure pulse and deterministic travel delay equal to the zone-index distance from origin to cut. Each cut bins traversing flow by that idealized arrival delay. At unit width, the arrival bin identifies the origin. Consecutive origin-resolved cut counts then recover the flow ending at each destination by subtraction.

|Bin width|Rank added beyond OD margins across ten networks|Remaining linear nullity|
|---:|---:|---:|
|1|561–4,851|0|
|8|60–576|501–4,275|
|16|22–264|539–4,587|
|32|3–108|558–4,743|

Seed 2026110342 fixes one integer-flow reconstruction check per network. The independent checker streams each path's contributions into bins, confirms exact recovery at unit width and uses SVD on independently assembled coarse operators. The portable run matches all 40 results and input hashes.

The experiment specifies the information an idealized timing model would provide. No available detector has been shown to follow that model. Zone-index delay is a design assumption. It is not a measured travel time or a verified alignment between released tasks.

```powershell
python task4/audit_hypothetical_time_bins.py --release-root <release-root> --output <new-output-directory>
```
