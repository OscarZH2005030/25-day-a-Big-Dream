# Transfer of covariance shape on released count residuals

Fit covariance shape to train counts minus the projected weak prior and evaluate the selected shape on validation residuals. Freeze variance weights to the train prior. Profile one scalar scale per residual vector. Score raw residuals and contrasts that remove the prior and linear OD span count profiles.

Among the 18 prespecified candidates all roads and both views chose power zero, shared fraction 0.5, and equal origin and destination shares. Validation scores improved over the fixed square-root baseline by 0.128 and 0.146 per coordinate, respectively. The grid has no diagonal model and shared fraction 0.5 is its upper boundary. The score is an observable proxy based on released residuals and independent count rows. It does not validate error generation or hidden OD accuracy.

    python task4/audit_released_residual_shape_transfer.py --release-root /path/to/release --output /path/to/output
