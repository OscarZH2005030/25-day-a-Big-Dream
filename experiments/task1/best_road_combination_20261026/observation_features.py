import numpy as np
import pandas as pd
CHANNELS=['speed_kmh','flow_vph']

def feature_grid_from_observations(self, date, observations):
        x = observations
        wd = pd.Timestamp(date).dayofweek
        hist = self.history[:, :, wd]
        features = {}
        base = []
        ticks = np.broadcast_to(np.arange(288), (self.n, 288))
        features['weekday'] = np.full(ticks.shape, wd)
        features['slot_sin'] = np.sin(2*np.pi*ticks/288)
        features['slot_cos'] = np.cos(2*np.pi*ticks/288)
        for col in ['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph']:
            features[col] = np.broadcast_to(self.static[col].to_numpy()[:, None], ticks.shape)
        for c, col in enumerate(CHANNELS):
            a = self.grid(x, col)
            if c == 1:
                a = a/self.lanes[:, None]
            valid = np.isfinite(a)
            prev = np.maximum.accumulate(np.where(valid, ticks, -1), axis=1)
            nxt = np.minimum.accumulate(np.where(valid, ticks, 288)[:, ::-1], axis=1)[:, ::-1]
            pv = np.take_along_axis(a, np.clip(prev, 0, 287), axis=1)
            nv = np.take_along_axis(a, np.clip(nxt, 0, 287), axis=1)
            pv[prev<0] = np.nan
            nv[nxt>287] = np.nan
            both = (prev>=0) & (nxt<288)
            fraction = (ticks-prev)/np.maximum(nxt-prev, 1)
            # Outside the observed span use history; avoid propagating a distant
            # endpoint indefinitely. Within the span use linear interpolation.
            initial = np.where(both, pv+(nv-pv)*fraction, hist[c])
            base.append(initial)
            features[col+'_history'] = hist[c]
            features[col+'_initial'] = initial
            features[col+'_prev'] = pv
            features[col+'_next'] = nv
            features[col+'_prev_gap_min'] = np.where(prev>=0, (ticks-prev)*5, np.nan)
            features[col+'_next_gap_min'] = np.where(nxt<288, (nxt-ticks)*5, np.nan)
            features[col+'_history_fallback'] = (~both).astype(float)
            for offset in [-6, -3, -1, 1, 3, 6]:
                b = np.full_like(a, np.nan)
                if offset>0:
                    b[:, :-offset] = a[:, offset:]
                else:
                    b[:, -offset:] = a[:, :offset]
                features[f'{col}_offset_{offset*5}min'] = b
            for side, groups in self.neighbors.items():
                neighbor = np.full_like(a, np.nan)
                available = np.zeros_like(a)
                for i, ids in enumerate(groups):
                    if ids:
                        z = a[ids]
                        count = np.isfinite(z).sum(axis=0)
                        neighbor[i] = np.divide(np.nansum(z, axis=0), count,
                                                out=np.full(288, np.nan), where=count>0)
                        available[i] = count
                features[f'{col}_{side}'] = neighbor
                features[f'{col}_{side}_count'] = available
        for col in ['occupancy', 'pct_observed', 'is_imputed']:
            a = self.grid(x, col)
            # Contemporaneous target occupancy is blank; shifted values provide
            # usable surrounding evidence. Quality flags themselves are released.
            if col != 'occupancy':
                features[col] = a
            for offset in [-1, 1]:
                b = np.full_like(a, np.nan)
                if offset>0:
                    b[:, :-1] = a[:, 1:]
                else:
                    b[:, 1:] = a[:, :-1]
                features[f'{col}_offset_{offset*5}min'] = b
        features['relative_speed_initial'] = base[0]/self.static.free_speed_kmh.to_numpy()[:, None]
        features['capacity_utilization_initial'] = base[1]*self.lanes[:, None]/self.static.capacity_vph.to_numpy()[:, None]
        return features, base, hist, x
