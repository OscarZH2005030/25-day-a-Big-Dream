# Synthetic allocation recovery with a biased mean

Reuse the same 92,160 Gaussian draws and candidate grid. Add mean error along the pooled-prior count profile at three prespecified amplitudes. The largest amplitude has whitened RMS one per independent count coordinate under the generating model. Compare the raw likelihood with likelihood in a contrast space orthogonal to the known direction.

The average accuracy falls from 97.15% to 79.36% at the highest shift without adjustment. With the known direction removed it is 96.97% at all three amplitudes. This tests a specified synthetic mean error and does not verify that real mean errors follow that direction.

    python task4/audit_synthetic_allocation_center_bias.py --release-root /path/to/release --output /path/to/output
