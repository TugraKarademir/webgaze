"""
analyze.py - calib_data.npz uzerinde farkli ozellik/model ayarlarini karsilastirir.
Kalibrasyon verisiyle egitir, dogrulama noktalarinda olcer (yeniden kalibrasyon gerekmez).

  python analyze.py [calib_data.npz]
"""
import itertools
import sys

import numpy as np

import webcam_eye_tracker as w

MM_PX = w.ctypes.windll.gdi32.GetDeviceCaps(w.user32.GetDC(0), 4) / w.SCREEN_W  # piksel boyu (mm)


# Ham sutunlar (webcam_eye_tracker.model_features aciklamasina bakin)
def feats_avg(R):
    return np.column_stack([(R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2, (R[:, 4] + R[:, 5]) / 2,
                            R[:, 6] / 30, R[:, 7] / 30, R[:, 9], R[:, 10], R[:, 11]])


def feats_per_eye(R):
    return w.model_features(R)


def feats_eyes_only(R):
    return np.column_stack([(R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2, (R[:, 4] + R[:, 5]) / 2])


def feats_no_pose(R):
    return np.column_stack([(R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2, (R[:, 4] + R[:, 5]) / 2,
                            R[:, 9], R[:, 10], R[:, 11]])


def feats_per_eye_only(R):
    return np.column_stack([R[:, 0], R[:, 1], R[:, 2], R[:, 3], (R[:, 4] + R[:, 5]) / 2])


def feats_eyes_pos(R):
    return np.column_stack([(R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2, (R[:, 4] + R[:, 5]) / 2,
                            R[:, 9], R[:, 10]])


AVG_COLS, PER_EYE_COLS = ([0], [1]), ([0, 2], [1, 3])
FEATS = {"ortalama+bas": (feats_avg, AVG_COLS), "ayri goz+bas": (feats_per_eye, PER_EYE_COLS),
         "sadece goz": (feats_eyes_only, AVG_COLS), "goz+konum (aci yok)": (feats_no_pose, AVG_COLS),
         "ayri goz (bas yok)": (feats_per_eye_only, PER_EYE_COLS), "goz+burun": (feats_eyes_pos, AVG_COLS)}


def smooth(R, pid, k):
    """Her nokta icinde k karelik nedensel hareketli ortalama (canli takipte de uygulanabilir)."""
    if k <= 1:
        return R
    out = R.copy()
    for p in np.unique(pid):
        idx = np.where(pid == p)[0]
        X = R[idx]
        c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
        n = np.arange(1, len(X) + 1)
        lo = np.maximum(0, n - k)
        out[idx] = (c[n] - c[lo]) / (n - lo)[:, None]
    return out


def evaluate(d, feat_cols, lam, k, outliers):
    feat, w.EYE_COLS = feat_cols
    cR, cY, cp = d["calib_R"][:, 1:], d["calib_Y"], d["calib_pid"]
    vR, vY, vp = d["valid_R"][:, 1:], d["valid_Y"], d["valid_pid"]
    cR, vR = smooth(cR, cp, k), smooth(vR, vp, k)
    F, Y = [], []
    for p in np.unique(cp):
        f = feat(cR[cp == p])
        if outliers:
            keep = w.drop_outliers(np.column_stack([f[:, :2], np.arange(len(f))]))[:, -1].astype(int)
            f = f[keep]
        F.append(f)
        Y.append(np.tile(cY[cp == p][0], (len(f), 1)))
    m = w.GazeModel()
    m.fit(np.vstack(F), np.vstack(Y), lam=lam)
    acc, prec = [], []
    for p in np.unique(vp):
        P = m.predict(feat(vR[vp == p]))
        acc.append(np.linalg.norm(P.mean(0) - vY[vp == p][0]))
        prec.append(np.linalg.norm(P - P.mean(0), axis=1).mean())
    return float(np.mean(acc)), float(np.mean(prec))


def main():
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else "calib_data.npz")
    print(f"kalibrasyon ornegi {len(d['calib_R'])}, dogrulama ornegi {len(d['valid_R'])}")
    rows = []
    for (name, feat), lam, k, out in itertools.product(FEATS.items(), [0.1, 1, 10, 100], [1, 3, 6, 10], [True]):
        acc, prec = evaluate(d, feat, lam, k, out)
        rows.append((acc, prec, name, lam, k))
    rows.sort()
    print(f"{'hata px':>8} {'cm':>5} {'titreme':>8}  ozellik               lambda  pencere")
    for acc, prec, name, lam, k in rows[:15]:
        print(f"{acc:8.0f} {acc * MM_PX / 10:5.1f} {prec:8.0f}  {name:22s} {lam:6g}  {k:4d}")
    old = evaluate(d, FEATS["ortalama+bas"], 1, 1, True)
    new = evaluate(d, FEATS["ayri goz (bas yok)"], 10, w.SMOOTH, True)
    print(f"\neski ayar: hata {old[0]:.0f} px, titreme {old[1]:.0f} px")
    print(f"yeni ayar: hata {new[0]:.0f} px, titreme {new[1]:.0f} px")


if __name__ == "__main__":
    main()
