# Active forward candidate and public input checks

The entry point regenerates the 70,708 prediction rows used by Jayden commit 35ed471. The local regeneration matches that export byte for byte. The isolated entry point has identical keys and input hashes, with a maximum flow difference of 2.31e-13.

Against the frozen prediction, mean published count NMAE is 0.006059% for validation and 0.007677% for private for both versions. Relative to the geometry-only cross-fit, the pair feature lowers held-out public weak-prior MSE by a mean of 16.34% and 18.49%, respectively. These are exposed public inputs. The weak-prior metric measures prediction of supplied prior entries.

```sh
python task4/export_forward_candidate.py --release-root /path/to/kaggle_public --output /path/to/output
```
