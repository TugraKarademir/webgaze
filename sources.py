"""
sources.py - Goruntu kaynaklari: siradan web kamerasi ya da Intel RealSense (RGB veya IR + derinlik).

Her kaynak read() ile (BGR goruntu, derinlik dizisi | None) dondurur; point_3d() bir pikselin
kameraya gore 3B konumunu (mm) verir (derinlik yoksa None).
"""
import cv2
import numpy as np


class WebcamSource:
    def __init__(self, index, size, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # Dahili kamerada 1280x720 sadece 10 fps, 640x480 ise 30 fps veriyordu; MJPG ile hiz korunur
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera {index} acilamadi")
        self.name = f"web kamerasi {index}"

    def read(self):
        ok, frame = self.cap.read()
        return (frame, None) if ok else (None, None)

    def focal(self, w):
        return w  # bilinmiyor: ~53 derece yatay goruş acisi varsayimi

    def point_3d(self, depth, px, py):
        return None

    def close(self):
        self.cap.release()


class RealSenseSource:
    """
    stream="color": RGB goruntu, derinlik renk kamerasina hizalanir.
    stream="ir":     sol kizilotesi kamera. Derinlik zaten bu kameranin bakis acisinda oldugu icin
                     hizalama gerekmez. Kizilotesi goruntu ortam isigindan az etkilenir ve goz bebegi
                     daha keskin gorunur; projektorun nokta deseni yuze dusmesin diye varsayilan olarak
                     kapatilir (emitter=False), bu da derinligi biraz zayiflatir.
    """

    def __init__(self, size, fps=30, stream="color", emitter=None):
        import pyrealsense2 as rs
        self.rs = rs
        if not rs.context().devices:
            raise RuntimeError("Bagli RealSense cihazi bulunamadi (USB 3 portuna takili mi?)")
        self.stream = stream
        w, h = size
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        if stream == "ir":
            cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
        else:
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        dev = prof.get_device()
        sensor = dev.first_depth_sensor()
        self.scale = sensor.get_depth_scale()  # derinlik birimi -> metre
        if emitter is None:
            emitter = stream != "ir"
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1 if emitter else 0)
        vs = (prof.get_stream(rs.stream.infrared, 1) if stream == "ir" else prof.get_stream(rs.stream.color))
        self.intr = vs.as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color) if stream == "color" else None
        self.name = f"{dev.get_info(rs.camera_info.name)} ({'IR' if stream == 'ir' else 'RGB'} + derinlik)"

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(1000)
        except RuntimeError:
            return None, None
        if self.align:
            frames = self.align.process(frames)
        img = frames.get_infrared_frame(1) if self.stream == "ir" else frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not img or not depth:
            return None, None
        a = np.asanyarray(img.get_data())
        if a.ndim == 2:  # MediaPipe 3 kanal bekler
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        return a, np.asanyarray(depth.get_data()).copy()

    def focal(self, w):
        return self.intr.fx

    def point_3d(self, depth, px, py, r=3):
        """Pikselin etrafindaki kucuk pencerenin medyan derinligiyle 3B konum (mm, kamera eksenleri)."""
        x, y = int(round(px)), int(round(py))
        patch = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        z = float(np.median(patch)) * self.scale
        return np.array(self.rs.rs2_deproject_pixel_to_point(self.intr, [float(px), float(py)], z)) * 1000.0

    def close(self):
        self.pipe.stop()


def realsense_connected():
    try:
        import pyrealsense2 as rs
        return len(rs.context().devices) > 0
    except Exception:
        return False


def list_cameras():
    """(indeks, ad) listesi. Indeksler OpenCV/DirectShow siralamasidir (--camera ile verilen)."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        return [(c.index, c.name) for c in enumerate_cameras(cv2.CAP_DSHOW)]
    except ImportError:
        return []


def resolve_camera(camera):
    """--camera: sayi ya da adin bir parcasi ('integrated', 'C920' ...) -> indeks."""
    if camera is None:
        return 0
    if str(camera).isdigit():
        return int(camera)
    cams = list_cameras()
    for idx, name in cams:
        # RealSense'in DirectShow 'Depth' ucu siradan goruntu vermez, ad aramasinda atla
        if str(camera).lower() in name.lower() and not name.lower().rstrip().endswith("depth"):
            return idx
    names = ", ".join(n for _, n in cams) or "liste alinamadi"
    raise RuntimeError(f"'{camera}' adinda kamera bulunamadi. Bagli kameralar: {names}")


def print_cameras():
    print("Web kameralari (--camera ile secin, sayi ya da adin bir parcasi):")
    for idx, name in list_cameras():
        note = "  (RealSense icin --realsense kullanin)" if "realsense" in name.lower() else ""
        print(f"  {idx}: {name}{note}")
    print("RealSense:", "bagli" if realsense_connected() else "bagli degil")


def open_source(args):
    size = (args.width, args.height)
    if args.realsense:
        emitter = True if args.rs_emitter else None
        return RealSenseSource(size, args.fps, args.rs_stream, emitter)
    return WebcamSource(resolve_camera(args.camera), size, args.fps)
