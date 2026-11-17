"""
webcam_eye_tracker.py - Tobii benzeri goz takibi, sadece web kamerasi ile (Windows).

Ne yapar:
  * MediaPipe Face Mesh ile iris + goz kosesi + bas pozu (head pose) cikarir
  * 13 noktali kalibrasyon -> ekran koordinatina regresyon (ridge, 2. derece)
  * 9 noktali dogrulama: hata (px/derece), hassasiyet ve bolge isabet orani raporu
  * Ilgi alanlari (--aoi): bakis en yakin bolgeye oturtulur, CSV'ye bolge adi yazilir
  * Tiklamalarla surekli kalibrasyon: tiklanan yere bakildigi varsayilip model guncellenir
  * Ekranin ustunde tiklanamaz (click-through) bakis noktasi gostergesi
  * Istege bagli: bakisla fare kontrolu, CSV kaydi, OpenTrack'e bas takibi (UDP)

Kisayollar (her yerde calisir):
  F5  tum bolgelerin cercevesini goster/gizle
  F6  bas takibi sifirla (OpenTrack merkez)
  F7  bakis gostergesini ac/kapat
  F8  bakisla fare kontrolunu ac/kapat
  F9  yeniden kalibre et
  F10 cikis

Ornek:
  python webcam_eye_tracker.py --preview --log gaze.csv
  python webcam_eye_tracker.py --aoi grid:3x3 --log gaze.csv
  python webcam_eye_tracker.py --aoi auto          # ekrandaki ogeleri otomatik bul
  python webcam_eye_tracker.py --aoi aoi.json      # elle tanimli bolgeler
  python webcam_eye_tracker.py --load calib.npz --opentrack
"""
import argparse
import collections
import csv
import ctypes
import ctypes.wintypes
import math
import random
import socket
import struct
import threading
import time
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np

import regions as aoi

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # gercek piksel koordinatlari
except Exception:
    user32.SetProcessDPIAware()
SCREEN_W, SCREEN_H = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# MediaPipe landmark indeksleri (goruntude soldan saga: a -> b)
EYE_R = dict(a=33, b=133, top=159, bot=145, iris=[468, 469, 470, 471, 472])
EYE_L = dict(a=362, b=263, top=386, bot=374, iris=[473, 474, 475, 476, 477])
# Bas pozu icin genel 3B yuz modeli (x saga, y asagi, z kameradan uzaga)
POSE_IDX = [1, 152, 263, 33, 291, 61]
MODEL_3D = np.array([(0, 0, 0), (0, 330, 65), (225, -170, 135), (-225, -170, 135),
                     (150, 150, 125), (-150, 150, 125)], dtype=np.float64)

SETTLE, COLLECT = 0.7, 1.0  # kalibrasyon noktasi basina saniye

# Tiklamayla ogrenme: goz tiklamadan once hedefe varir, bu yuzden tiklamadan
# 350-50 ms onceki ornekler o noktaya bakarken alinmis kabul edilir.
CLICK_WINDOW = (0.35, 0.05)
CLICK_WEIGHT = 3   # her tiklama ornegi kalibrasyon ornegine gore bu kadar agirlikli
CLICK_MAX = 600    # en fazla bu kadar tiklama ornegi tutulur (eskiler duser)


def eye_features(p, e):
    a, b = p[e["a"]], p[e["b"]]
    c = p[e["iris"]].mean(axis=0)
    w = np.linalg.norm(b - a)
    ux = (b - a) / w
    uy = np.array([-ux[1], ux[0]])
    d = c - (a + b) / 2
    return d @ ux / w, d @ uy / w, np.linalg.norm(p[e["top"]] - p[e["bot"]]) / w


def euler_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    pitch = math.atan2(R[2, 1], R[2, 2])
    roll = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class OneEuro:
    """Gurultuyu azaltir ama hizli bakis hareketlerinde gecikme yapmaz."""

    def __init__(self, mincutoff=0.6, beta=0.004, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.t = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._alpha(self.dcutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(self.mincutoff + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


# Ham ozellik vektoru (Tracker uretir, calib_data.npz'ye kaydedilir):
#  0 rx, 1 ry, 2 lx, 3 ly  sag/sol iris konumu (goz genisligine gore)
#  4 ro, 5 lo              goz aciklik orani
#  6 yaw, 7 pitch, 8 roll  bas acisi (derece)
#  9 nx, 10 ny, 11 fw      burun konumu ve yuz genisligi (goruntuye gore)
#  12 tx, 13 ty, 14 tz     bas konumu (model birimi)
def model_features(R):
    # iki goz ayri, bas acisi yok: analyze.py ile iki kayitta da en dusuk hata bu secimde cikti
    # (bas acisi tahmini bir bakis noktasi icinde ~2.6 derece titriyor, modele gurultu ekliyordu)
    R = np.atleast_2d(R)
    return np.column_stack([R[:, 0], R[:, 1], R[:, 2], R[:, 3], (R[:, 4] + R[:, 5]) / 2])


SMOOTH = 3  # ozellikler bu kadar karenin ortalamasi alinarak modele verilir (30 fps'de ~70 ms)


def smooth_rows(R, k=SMOOTH):
    """Nedensel hareketli ortalama: her satir kendisi ve onceki k-1 satirin ortalamasi."""
    c = np.cumsum(np.vstack([np.zeros((1, R.shape[1])), R]), axis=0)
    n = np.arange(1, len(R) + 1)
    lo = np.maximum(0, n - k)
    return (c[n] - c[lo]) / (n - lo)[:, None]


def drop_outliers(F, k=3.0):
    """Bir kalibrasyon noktasindaki sapan ornekleri (iris konumuna gore) at."""
    G = F[:, :2]
    med = np.median(G, axis=0)
    mad = np.median(np.abs(G - med), axis=0) * 1.4826 + 1e-6
    return F[(np.abs(G - med) / mad <= k).all(axis=1)]


EYE_COLS = ([0, 2], [1, 3])  # model_features icinde yatay / dikey iris sutunlari


def expand(F):
    F = np.atleast_2d(F)
    gx, gy = F[:, EYE_COLS[0]].mean(1), F[:, EYE_COLS[1]].mean(1)
    return np.column_stack([F, gx * gx, gy * gy, gx * gy])


class GazeModel:
    F = Y = None  # kalibrasyon ornekleri (tiklamalarla yeniden egitim icin saklanir)

    def fit(self, F, Y, lam=10.0):
        self.F, self.Y = F, Y
        X = expand(F)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = self._design(X)
        reg = lam * np.eye(Xs.shape[1])
        reg[0, 0] = 0
        self.W = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)

    def _design(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])

    def predict(self, f):
        return self._design(expand(f)) @ self.W

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, F=self.F, Y=self.Y, screen=[SCREEN_W, SCREEN_H])

    def load(self, path):
        d = np.load(path)
        self.mu, self.sd, self.W = d["mu"], d["sd"], d["W"]
        if "F" in d.files:
            self.F, self.Y = d["F"], d["Y"]
        if len(self.mu) != expand(model_features(np.zeros(15))).shape[1]:
            raise SystemExit(f"{path} eski surume ait, --load olmadan yeniden kalibre edin.")


class Tracker(threading.Thread):
    """Kamera okur, her karede ozellik vektoru + bas pozu uretir."""

    def __init__(self, cam, preview, blink_thr, udp, size=(640, 480)):
        super().__init__(daemon=True)
        self.cam, self.preview, self.blink_thr, self.size = cam, preview, blink_thr, size
        self.udp = udp
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp else None
        self.lock = threading.Lock()
        self.latest = None  # (t, feat, blink)
        self.rec = None
        self.pose_ref = None
        self.running = True
        self.error = None

    def start_rec(self):
        with self.lock:
            self.rec = []

    def stop_rec(self):
        with self.lock:
            r, self.rec = self.rec, None
        return r

    def get_latest(self):
        with self.lock:
            return self.latest

    def recenter(self):
        self.pose_ref = None

    def run(self):
        cap = cv2.VideoCapture(self.cam, cv2.CAP_DSHOW)
        # Bu kamerada 1280x720 sadece 10 fps, 640x480 ise 30 fps veriyor
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            self.error = f"Kamera {self.cam} acilamadi"
            print(self.error)
            return
        mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        rvec = tvec = None
        fps_t, fps_n, self.fps = time.perf_counter(), 0, 0.0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.perf_counter()
            fps_n += 1
            if t - fps_t >= 2:
                first = self.fps == 0
                self.fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0
                if first:
                    print(f"Kamera: {frame.shape[1]}x{frame.shape[0]}, {self.fps:.0f} fps")
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                p = np.array([(q.x * w, q.y * h) for q in lm])
                rx, ry, ro = eye_features(p, EYE_R)
                lx, ly, lo = eye_features(p, EYE_L)
                openness = (ro + lo) / 2

                K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
                img_pts = p[POSE_IDX].astype(np.float64)
                if rvec is None:
                    _, rvec, tvec = cv2.solvePnP(MODEL_3D, img_pts, K, None, flags=cv2.SOLVEPNP_EPNP)
                _, rvec, tvec = cv2.solvePnP(MODEL_3D, img_pts, K, None, rvec, tvec,
                                             useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                yaw, pitch, roll = euler_deg(rvec)
                face_w = np.linalg.norm(p[263] - p[33]) / w

                raw = np.array([rx, ry, lx, ly, ro, lo, yaw, pitch, roll,
                                p[1, 0] / w - 0.5, p[1, 1] / h - 0.5, face_w, *tvec.ravel()])
                blink = openness < self.blink_thr
                with self.lock:
                    self.latest = (t, raw, blink)
                    if self.rec is not None and not blink:
                        self.rec.append(np.r_[t, raw])

                if self.sock:
                    # OpenTrack "UDP over network": x,y,z (cm), yaw,pitch,roll (derece)
                    pose = np.array([*(tvec.ravel() / 50.0), yaw, pitch, roll])
                    if self.pose_ref is None:
                        self.pose_ref = pose
                    self.sock.sendto(struct.pack("<6d", *(pose - self.pose_ref)), self.udp)

                if self.preview:
                    for e in (EYE_R, EYE_L):
                        cx, cy = p[e["iris"]].mean(axis=0)
                        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                        for k in ("a", "b", "top", "bot"):
                            cv2.circle(frame, tuple(p[e[k]].astype(int)), 2, (0, 200, 255), -1)
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"{self.fps:.0f} fps  yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f}"
                                       f"{'  GOZ KIRPMA' if blink else ''}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif self.preview:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "Yuz bulunamadi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if self.preview:
                cv2.imshow("Kamera", cv2.resize(frame, (640, int(640 * h / w))))
                cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()


def make_click_through(win):
    hwnd = user32.GetParent(win.winfo_id())
    style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
    user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80)  # LAYERED|TRANSPARENT|TOOLWINDOW


class App:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.withdraw()
        udp = ("127.0.0.1", args.opentrack_port) if args.opentrack else None
        self.tracker = Tracker(args.camera, args.preview, args.blink, udp, (args.width, args.height))
        self.tracker.start()
        self.model = GazeModel()
        self.ov = None
        self.show_dot = True
        self.mouse = False
        self.tracking = False
        self.keys = {}
        self.last_t = None
        self.fx, self.fy = OneEuro(), OneEuro()
        self.regions, self.auto = aoi.parse_spec(args.aoi, SCREEN_W, SCREEN_H) if args.aoi else ([], None)
        self.assigner = aoi.RegionAssigner(snap=args.snap)
        self.show_all_regions = False
        self.hist = collections.deque(maxlen=60)  # (t, ozellik, tahmin): tiklama aninda geriye bakmak icin
        self.click_F = collections.deque(maxlen=CLICK_MAX)
        self.click_Y = collections.deque(maxlen=CLICK_MAX)
        self.log = None
        if args.log:
            f = open(args.log, "w", newline="", encoding="utf-8", buffering=1)  # pencere kapatilsa da veri kalsin
            self.log = (f, csv.writer(f))
            self.log[1].writerow(["t", "raw_x", "raw_y", "x", "y", "blink", "aoi"])
        if args.load:
            self.model.load(args.load)
            print(f"Kalibrasyon yuklendi: {args.load}")
            self.start_tracking()
        else:
            self.calibrate()
        self.root.after(30, self.poll_keys)

    # ---------- kalibrasyon ----------
    def calibrate(self):
        self.tracking = False
        if self.ov:
            self.ov.withdraw()
        win = tk.Toplevel(self.root)
        win.attributes("-fullscreen", True)
        win.attributes("-topmost", True)
        c = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0, cursor="none")
        c.pack(fill="both", expand=True)
        win.focus_force()
        self.cwin, self.cc = win, c
        self.ctext = c.create_text(SCREEN_W / 2, SCREEN_H / 2, fill="white", font=("Segoe UI", 20), justify="center")
        win.bind("<space>", lambda e: self._calib_start())
        win.bind("<Escape>", lambda e: self.quit())
        self._waiting = True
        self._face_status()

    def _face_status(self):
        if not self._waiting:
            return
        lt = self.tracker.get_latest()
        ok = lt is not None and time.perf_counter() - lt[0] < 0.5
        msg = self.tracker.error or ("Yuz algilandi ✓" if ok else "Yuz bulunamadi - kameraya bakin")
        self.cc.itemconfigure(self.ctext, text=(
            "KALIBRASYON\n\nBasinizi sabit tutun, sadece gozlerinizle kirmizi noktalari takip edin.\n"
            f"\n{msg}\n\nBaslamak icin BOSLUK, cikmak icin ESC"))
        self.root.after(200, self._face_status)

    def _calib_start(self):
        if not self._waiting:
            return
        self._waiting = False
        self.cwin.unbind("<space>")
        self.cc.itemconfigure(self.ctext, text="")
        m = 0.06
        pts = [(x, y) for y in (m, 0.5, 1 - m) for x in (m, 0.5, 1 - m)]
        pts += [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)]
        random.shuffle(pts)
        self._run_points(pts, self._calib_done)

    def _run_points(self, points, done):
        data = []
        c = self.cc

        def step(i):
            if i == len(points):
                c.delete("dot")
                done(data)
                return
            x, y = points[i][0] * SCREEN_W, points[i][1] * SCREEN_H
            t0 = time.perf_counter()
            state = {"rec": False}

            def anim():
                el = time.perf_counter() - t0
                r = max(5.0, 28 - 23 * el / SETTLE)
                c.delete("dot")
                c.create_oval(x - r, y - r, x + r, y + r, fill="#ff3b3b", outline="", tags="dot")
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="white", outline="", tags="dot")
                if el >= SETTLE and not state["rec"]:
                    state["rec"] = True
                    self.tracker.start_rec()
                if el >= SETTLE + COLLECT:
                    data.append((points[i], np.array(self.tracker.stop_rec())))
                    step(i + 1)
                else:
                    self.root.after(16, anim)

            anim()

        step(0)

    def _calib_done(self, data):
        if any(len(f) < 5 for _, f in data):
            self._calib_fail("Bazi noktalarda yuz/goz algilanamadi.")
            return
        self._calib_data = data
        clean = [(p, drop_outliers(model_features(smooth_rows(r[:, 1:])))) for p, r in data]
        F = np.vstack([f for _, f in clean])
        Y = np.vstack([np.tile([p[0] * SCREEN_W, p[1] * SCREEN_H], (len(f), 1)) for p, f in clean])
        self.model.fit(F, Y)
        self.click_F.clear()
        self.click_Y.clear()
        # 3x3 hucrenin her birinde rastgele bir nokta (hucre merkezine sabit degil, kenara yakin da olabilir):
        # bolge isabet orani boylece iyimser olmadan olculur
        vpts = [((c + random.uniform(0.15, 0.85)) / 3, (r + random.uniform(0.15, 0.85)) / 3)
                for r in range(3) for c in range(3)]
        random.shuffle(vpts)
        self._run_points(vpts, self._valid_done)

    def _save_raw(self, valid):
        """Ham kalibrasyon/dogrulama verisi: farkli modelleri cevrimdisi denemek icin."""
        def pack(data):
            R = np.vstack([r for _, r in data])
            Y = np.vstack([np.tile([p[0] * SCREEN_W, p[1] * SCREEN_H], (len(r), 1)) for p, r in data])
            pid = np.concatenate([np.full(len(r), i) for i, (_, r) in enumerate(data)])
            return R, Y, pid
        cR, cY, cp = pack(self._calib_data)
        vR, vY, vp = pack(valid)
        np.savez("calib_data.npz", calib_R=cR, calib_Y=cY, calib_pid=cp,
                 valid_R=vR, valid_Y=vY, valid_pid=vp, screen=[SCREEN_W, SCREEN_H])

    def _calib_fail(self, why):
        self.cc.itemconfigure(self.ctext, text=f"{why}\n\nTekrar denemek icin BOSLUK")
        self._waiting = True
        self.cwin.bind("<space>", lambda e: self._calib_start())

    def _valid_done(self, data):
        if any(len(f) < 5 for _, f in data):
            self._calib_fail("Dogrulama sirasinda goz algilanamadi.")
            return
        acc, prec, prec_f, allP, allT = [], [], [], [], []
        for p, r in data:
            P = self.model.predict(model_features(smooth_rows(r[:, 1:])))
            target = np.array([p[0] * SCREEN_W, p[1] * SCREEN_H])
            allP.append(P)
            allT.append(np.tile(target, (len(P), 1)))
            acc.append(np.linalg.norm(P.mean(0) - target))
            prec.append(np.linalg.norm(P - P.mean(0), axis=1).mean())
            # ekranda gorulen (One Euro filtreli) titreme
            fx, fy = OneEuro(), OneEuro()
            Q = np.array([(fx(x, t), fy(y, t)) for t, (x, y) in zip(r[:, 0], P)])[len(P) // 3:]
            prec_f.append(np.linalg.norm(Q - Q.mean(0), axis=1).mean())
        acc, prec, prec_f = float(np.mean(acc)), float(np.mean(prec)), float(np.mean(prec_f))
        self.model.save(self.args.save)
        self._save_raw(data)
        mm_px = ctypes.windll.gdi32.GetDeviceCaps(user32.GetDC(0), 4) / SCREEN_W  # piksel boyu (mm)
        deg = math.degrees(math.atan(acc * mm_px / (10 * self.args.distance)))
        report = (f"Dogrulama: ortalama hata {acc:.0f} px = {acc * mm_px / 10:.1f} cm = "
                  f"~{deg:.1f} derece ({self.args.distance:.0f} cm uzaklikta), "
                  f"hassasiyet {prec:.0f} px ham / {prec_f:.0f} px filtreli")
        allP, allT = np.vstack(allP), np.vstack(allT)
        hits = [f"{n}x{n}: %{100 * aoi.hit_rate(allP, allT, aoi.grid_regions(n, n, SCREEN_W, SCREEN_H)):.0f}"
                for n in (2, 3, 4)]
        report += "\nBolge isabeti  " + "   ".join(hits)
        if self.regions and not self.auto and not self.args.aoi.startswith("grid:"):
            report += f"   secili bolgeler: %{100 * aoi.hit_rate(allP, allT, self.regions, self.args.snap):.0f}"
        print(report, f"-> kaydedildi: {self.args.save}, calib_data.npz")
        self.cc.itemconfigure(self.ctext, text=(
            f"{report}\n\nKalibrasyon '{self.args.save}' dosyasina kaydedildi.\n\n"
            "Devam: BOSLUK     Yeniden kalibre: R"))
        self.cwin.bind("<space>", lambda e: self._close_calib())
        self.cwin.bind("r", lambda e: self._recal())

    def _recal(self):
        self.cwin.destroy()
        self.calibrate()

    def _close_calib(self):
        self.cwin.destroy()
        self.start_tracking()

    # ---------- takip ----------
    def start_tracking(self):
        if self.ov is None:
            ov = tk.Toplevel(self.root)
            ov.overrideredirect(True)
            ov.geometry(f"{SCREEN_W}x{SCREEN_H}+0+0")
            ov.attributes("-topmost", True)
            ov.configure(bg="magenta")
            ov.attributes("-transparentcolor", "magenta")
            c = tk.Canvas(ov, bg="magenta", highlightthickness=0)
            c.pack(fill="both", expand=True)
            self.dot = c.create_oval(0, 0, 0, 0, outline="#00d8ff", width=4)
            self.aoi_box = c.create_rectangle(0, 0, 0, 0, outline="#2ecc40", width=3, state="hidden")
            ov.update_idletasks()
            make_click_through(ov)
            self.ov, self.oc = ov, c
        if self.show_dot:
            self.ov.deiconify()
        self.fx, self.fy = OneEuro(), OneEuro()
        self.recent = collections.deque(maxlen=SMOOTH)
        self.tracking = True
        print("Takip basladi. F5 bolgeler | F6 bas merkez | F7 gosterge | F8 fare | F9 kalibrasyon | F10 cikis")
        if not self.args.no_click_learn and self.model.F is None:
            print("Not: bu kalibrasyon dosyasi eski, tiklamayla ogrenme icin yeniden kalibre edin.")
        self.gaze_loop()
        self.region_loop()

    def region_loop(self):
        """Otomatik bolgeleri al, F5 aciksa hepsinin cercevesini ciz."""
        if not self.tracking:
            return
        if self.auto:
            self.regions = self.auto.regions
        self.oc.delete("all_aoi")
        if self.show_all_regions:
            for r in self.regions:
                self.oc.create_rectangle(r.x, r.y, r.x + r.w, r.y + r.h, outline="#ffb000", width=1,
                                         dash=(4, 4), tags="all_aoi")
        self.root.after(500, self.region_loop)

    def gaze_loop(self):
        if not self.tracking:
            return
        lt = self.tracker.get_latest()
        if lt and lt[0] != self.last_t:
            t, raw, blink = lt
            self.last_t = t
            if not blink:
                self.recent.append(raw)
                f = model_features(np.mean(self.recent, axis=0))
                rx, ry = self.model.predict(f)[0]
                rx, ry = min(max(rx, 0), SCREEN_W - 1), min(max(ry, 0), SCREEN_H - 1)
                self.hist.append((t, f[0], (rx, ry)))
                x, y = self.fx(rx, t), self.fy(ry, t)
                r = 22
                self.oc.coords(self.dot, x - r, y - r, x + r, y + r)
                region = self.assigner(self.regions, x, y) if self.regions else None
                if region:
                    self.oc.coords(self.aoi_box, region.x, region.y, region.x + region.w, region.y + region.h)
                self.oc.itemconfigure(self.aoi_box, state="normal" if region else "hidden")
                if self.mouse:
                    user32.SetCursorPos(int(x), int(y))
                if self.log:
                    self.log[1].writerow([f"{t:.4f}", f"{rx:.1f}", f"{ry:.1f}", f"{x:.1f}", f"{y:.1f}", 0,
                                          region.name if region else ""])
            elif self.log:
                self.log[1].writerow([f"{t:.4f}", "", "", "", "", 1, ""])
        self.root.after(10, self.gaze_loop)

    # ---------- kisayollar ----------
    def _pressed(self, vk):
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was = self.keys.get(vk, False)
        self.keys[vk] = down
        return down and not was

    def learn_click(self):
        """Tiklanan noktaya bakildigini varsayip modeli bu ornekle yeniden egit."""
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        now = time.perf_counter()
        recent = [(f, p) for t, f, p in self.hist if now - CLICK_WINDOW[0] <= t <= now - CLICK_WINDOW[1]]
        if len(recent) < 2:
            return
        pred = np.mean([p for _, p in recent], axis=0)
        # tahmin tiklamadan cok uzaksa kisi muhtemelen baska yere bakarak tikladi: ogrenme
        if np.hypot(pred[0] - pt.x, pred[1] - pt.y) > self.args.click_radius:
            return
        for f, _ in recent:
            self.click_F.append(f)
            self.click_Y.append((pt.x, pt.y))
        cF = np.repeat(np.array(self.click_F), CLICK_WEIGHT, axis=0)
        cY = np.repeat(np.array(self.click_Y, dtype=float), CLICK_WEIGHT, axis=0)
        calF, calY = self.model.F, self.model.Y
        self.model.fit(np.vstack([calF, cF]), np.vstack([calY, cY]))
        self.model.F, self.model.Y = calF, calY  # kalibrasyon kumesi tiklamalarla sisirilmesin

    def poll_keys(self):
        if self._pressed(0x01) and self.tracking and not self.mouse and not self.args.no_click_learn \
                and self.model.F is not None:  # sol tik (bakisla fare acikken dongusel olur, atla)
            self.learn_click()
        if self._pressed(0x74):  # F5
            self.show_all_regions = not self.show_all_regions
        if self._pressed(0x75):  # F6
            self.tracker.recenter()
            print("Bas takibi merkezlendi")
        if self._pressed(0x76) and self.ov:  # F7
            self.show_dot = not self.show_dot
            (self.ov.deiconify if self.show_dot and self.tracking else self.ov.withdraw)()
        if self._pressed(0x77):  # F8
            self.mouse = not self.mouse
            print("Bakisla fare:", "ACIK" if self.mouse else "KAPALI")
        if self._pressed(0x78) and self.tracking:  # F9
            self.calibrate()
        if self._pressed(0x79):  # F10
            self.quit()
            return
        self.root.after(30, self.poll_keys)

    def quit(self):
        self.tracking = False
        self.tracker.running = False
        if self.auto:
            self.auto.running = False
        if self.click_F and self.model.F is not None:
            self.model.save(self.args.save)  # tiklamalarla iyilesen model sonraki --load icin kalsin
        if self.log:
            self.log[0].close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Web kamerasi ile goz takibi (Tobii benzeri)")
    ap.add_argument("--camera", type=int, default=0, help="kamera indeksi")
    ap.add_argument("--width", type=int, default=640, help="kamera genisligi")
    ap.add_argument("--height", type=int, default=480, help="kamera yuksekligi")
    ap.add_argument("--preview", action="store_true", help="kamera onizleme penceresi")
    ap.add_argument("--load", help="kayitli kalibrasyonu yukle (.npz)")
    ap.add_argument("--save", default="calib.npz", help="kalibrasyon kayit dosyasi")
    ap.add_argument("--log", help="bakis verisini CSV'ye yaz")
    ap.add_argument("--distance", type=float, default=45, help="gozden ekrana uzaklik (cm), derece hesabi icin")
    ap.add_argument("--aoi", help="ilgi alanlari: grid:3x3 | auto | bolgeler.json")
    ap.add_argument("--snap", type=float, default=80, help="bolge disina dusen bakisi bu kadar px icindeki bolgeye oturt")
    ap.add_argument("--no-click-learn", action="store_true", help="tiklamalarla surekli kalibrasyonu kapat")
    ap.add_argument("--click-radius", type=float, default=300,
                    help="tahmin tiklamaya bu kadar px'ten uzaksa o tiklamadan ogrenme")
    ap.add_argument("--blink", type=float, default=0.15, help="goz kirpma esigi (goz aciklik orani)")
    ap.add_argument("--opentrack", action="store_true", help="bas pozunu OpenTrack'e UDP ile gonder")
    ap.add_argument("--opentrack-port", type=int, default=4242)
    App(ap.parse_args()).run()


if __name__ == "__main__":
    main()
