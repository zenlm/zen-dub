"""Permissive landmark-based face mask (drop-in for the previous parser).

`FaceParsing.__call__(image, size, mode)` returns a single-channel PIL "L" mask
identical in spirit to the previous facial-region mask: white (255) over the
lower-face / mouth region to be re-rendered, black (0) elsewhere.

The mask source is the 2D-FAN-4 Face Alignment Network (1adrianb, BSD-3-Clause),
which is already vendored under ``musetalk/utils/face_detection``. The face crop
handed to this module by ``blending.get_image*`` is already a centred face, so the
68 landmarks are decoded directly from the heatmaps (argmax + quarter-pixel
sub-pixel refinement) and scaled into the mask resolution. From the landmarks we
build a synthetic face-region map (face oval from the jaw + lifted brow contour,
mouth from the convex hull of the mouth points) and then apply the same jaw /
neck morphology used downstream, so the composite produced by ``blending`` is
unchanged.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from ..face_detection.models import FAN
from ..face_detection.utils import get_preds_fromhm

# 2D-FAN-4 landmark weights (1adrianb / face-alignment, BSD-3-Clause).
FAN_WEIGHTS_URL = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"
# Optional local copy (populated by download_weights.sh); falls back to the URL above.
FAN_LOCAL_PATH = "./models/face-alignment/2DFAN4-cd938726ad.pth"


def _convex_hull_poly(points):
    """cv2.convexHull -> Nx2 float32 polygon (outer boundary of the point set)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def landmark_parse(landmarks_68, h, w, lift_ratio=0.30):
    """Synthetic face-region class map from 68 iBUG landmarks (in h x w pixel space).

    Emits only the classes the downstream morphology reads: ``1`` = facial skin
    (the face oval) and ``11`` = lips/mouth (always kept). The oval is the jaw
    contour (points 0..16) closed over the eyebrow points (17..26) lifted upward
    by ``lift_ratio * (jaw_bottom_y - brow_top_y)`` to reach the hairline; the
    lateral cone-dilation downstream supplies the remaining width, so the oval is
    not widened here. Mouth = convex hull of points 48..67.
    """
    lm = np.asarray(landmarks_68, dtype=np.float32)
    parsing = np.zeros((h, w), dtype=np.int32)

    jaw = lm[0:17]               # chin contour, left -> right
    brow = lm[17:27]             # eyebrows, left -> right
    jaw_bottom_y = float(jaw[:, 1].max())
    brow_top_y = float(brow[:, 1].min())
    span = max(jaw_bottom_y - brow_top_y, 1.0)
    lift = lift_ratio * span

    brow_lifted = brow.copy()
    brow_lifted[:, 1] -= lift
    oval = np.concatenate([jaw, brow_lifted[::-1]], axis=0)  # jaw L->R, brow R->L
    cv2.fillPoly(parsing, [np.round(oval).astype(np.int32)], 1)

    mouth_hull = _convex_hull_poly(lm[48:68])
    cv2.fillPoly(parsing, [np.round(mouth_hull).astype(np.int32)], 11)

    return parsing


class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80,
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._model_init()

        # --- jaw cone kernel (vertical chin extension), as before ---
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2
        for row in range(cone_height):
            if row < cone_height // 2:
                continue
            width = int(2 * (row - cone_height // 2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1
        base_width = int(kernel[cone_height - 1].sum()) if cone_height > 0 else 1
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width // 2)))
            end = min(total_size, int(center_x + (base_width // 2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel

        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
        self.cheek_mask = self._create_cheek_mask(left_cheek_width, right_cheek_width)

    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)
        return mask

    def _model_init(self):
        net = FAN(4)  # 2D-FAN-4 (4 stacked hourglasses)
        import os
        state = self._load_fan_state()
        net.load_state_dict(state)
        net.to(self.device)
        net.eval()
        return net

    @staticmethod
    def _load_fan_state():
        """Load the 2D-FAN-4 weights as a plain state_dict.

        The published face-alignment weight is distributed as a TorchScript
        archive; older mirrors ship a raw state_dict (optionally wrapped in a
        ``{"state_dict": ...}`` dict). All three forms are handled here.
        """
        import os
        path = FAN_LOCAL_PATH
        if not os.path.exists(path):
            from torch.hub import download_url_to_file, get_dir
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            cache = os.path.join(get_dir(), "checkpoints",
                                 os.path.basename(FAN_WEIGHTS_URL))
            if not os.path.exists(cache):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                download_url_to_file(FAN_WEIGHTS_URL, cache)
            path = cache
        try:  # TorchScript archive (current face-alignment distribution)
            return torch.jit.load(path, map_location="cpu").state_dict()
        except (RuntimeError, ValueError):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            return obj

    def _detect_landmarks(self, image, size):
        """68 landmarks (in ``size`` pixel space) for a centred face crop."""
        crop = image.convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(t)[-1].cpu()  # [1, 68, 64, 64]
        pts, _ = get_preds_fromhm(out)   # 64-grid coords, quarter-pixel refined
        pts = pts.squeeze(0).numpy()     # [68, 2]
        # 64-grid (1-based centres after get_preds_fromhm) -> size-space.
        out_w, out_h = size
        pts[:, 0] *= out_w / out.size(3)
        pts[:, 1] *= out_h / out.size(2)
        return pts

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, str):
            image = Image.open(image)

        try:
            landmarks = self._detect_landmarks(image, size)
        except Exception as exc:  # no face / detection failure -> no segment
            print("error, no person_segment:", exc)
            return None

        w, h = size
        parsing = landmark_parse(landmarks, h, w)

        # --- identical jaw / neck / raw morphology as the previous mask ---
        if mode == "neck":
            parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
            parsing[np.where(parsing != 255)] = 0
        elif mode == "jaw":
            face_region = (np.isin(parsing, [1]) * 255).astype(np.uint8)
            original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
            eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
            face_region = cv2.bitwise_and(eroded, self.cheek_mask)
            face_region = cv2.bitwise_or(
                face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
            parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
            parsing[np.isin(parsing, [11, 12, 13])] = 255
            parsing[np.where(parsing != 255)] = 0
        else:
            parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
            parsing[np.where(parsing != 255)] = 0

        return Image.fromarray(parsing.astype(np.uint8))


if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp("154_small.png")
    if segmap is not None:
        segmap.save("res.png")
