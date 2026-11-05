# Linear identifiability of the full OD span histogram

For all ten validation roads, count-only observations identify a one-dimensional space of linear histogram summaries, spanned by the total zone span. Adding hypothetical exact origin and destination totals expands this to two dimensions, spanned by the total number of trips and the total span. This is a row-space result. Zone span is the difference of the origin and destination zone indices.

    python task4/audit_pairing_span_histogram_geometry.py --release-root /path/to/release --output /path/to/output
