# Released OD pairing transfer by zone span

Evaluate the frozen half-average prediction in three equal-width zone-span bands on each road. After projecting out counts, margins and the span histogram, the shortest band represents a mean 81.0% of the released target projection's L1 mass. Its relative L1 error is 0.875, versus 0.785 and 0.788 in the middle and longest bands. The target is the released private weak prior, and zone span is an index difference.

    python task4/audit_pairing_prior_span_strata.py --release-root /path/to/release --output /path/to/output
