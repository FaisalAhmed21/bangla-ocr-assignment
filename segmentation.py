import numpy as np
from PIL import Image
from scipy import ndimage


class BanglaWordSegmenter:
    def __init__(
        self,
        merge_threshold_ratio: float = 0.015,
        min_area_ratio: float = 0.0005,
        padding_ratio: float = 0.18,
    ) -> None:
        self.merge_threshold_ratio = merge_threshold_ratio
        self.min_area_ratio = min_area_ratio
        self.padding_ratio = padding_ratio

    def segment(self, canvas_image: Image.Image) -> list[Image.Image]:
        binary = self._to_binary(canvas_image)
        if binary.sum() == 0:
            return []

        h, w = binary.shape
        min_area = max(12, int(self.min_area_ratio * h * w))

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not np.any(rows) or not np.any(cols):
            return []

        wy1, wy2 = np.where(rows)[0][[0, -1]]
        wx1, wx2 = np.where(cols)[0][[0, -1]]
        word_w = max(1, wx2 - wx1 + 1)
        word_h = max(1, wy2 - wy1 + 1)

        # First check if characters are already separate without cutting Matra
        labeled_orig, n_orig = ndimage.label(binary)
        boxes = self._extract_boxes(labeled_orig, n_orig, min_area, wy1, wy2, binary)
        boxes.sort(key=lambda b: b[0])
        boxes = self._merge_overlapping_boxes(boxes, gap=int(word_w * 0.03))

        # Only cut the top Matra (top 18% of word height) if the whole word is connected as 1 blob
        if len(boxes) <= 1 and word_w > word_h * 0.9:
            cut_binary = binary.copy()
            matra_y_end = wy1 + int(word_h * 0.18)
            for y in range(wy1, min(matra_y_end, h)):
                if cut_binary[y, wx1:wx2 + 1].sum() > (word_w * 0.35):
                    cut_binary[y, wx1:wx2 + 1] = False

            labeled, num_features = ndimage.label(cut_binary)
            boxes = self._extract_boxes(labeled, num_features, min_area, wy1, wy2, binary)
            boxes.sort(key=lambda b: b[0])
            boxes = self._merge_overlapping_boxes(boxes, gap=int(word_w * 0.03))

        if not boxes:
            boxes = [(int(wx1), int(wy1), int(wx2), int(wy2))]

        clean_rgb = np.where(binary[:, :, None], 0, 255).astype(np.uint8)
        clean_rgb = np.repeat(clean_rgb, 3, axis=2)

        return [self._crop_character(clean_rgb, box, h, w) for box in boxes]

    @staticmethod
    def _split_by_projection(
        cut_binary: np.ndarray, wx1: int, wy1: int, wx2: int, wy2: int, orig_binary: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        sub = cut_binary[wy1:wy2 + 1, wx1:wx2 + 1]
        col_sums = sub.sum(axis=0)
        word_w = wx2 - wx1 + 1
        word_h = wy2 - wy1 + 1

        # Smooth projection profile
        kernel_size = max(3, int(word_w * 0.03))
        smoothed = np.convolve(col_sums, np.ones(kernel_size) / kernel_size, mode="same")

        # Pixels with very low vertical ink density are gaps between characters
        threshold = max(1.5, np.percentile(smoothed, 25) * 0.6)
        is_ink = smoothed > threshold

        boxes = []
        in_char = False
        start_x = 0
        min_char_w = max(8, int(word_h * 0.18))

        for x in range(word_w):
            if is_ink[x] and not in_char:
                in_char = True
                start_x = x
            elif not is_ink[x] and in_char:
                in_char = False
                if x - start_x >= min_char_w:
                    cx1, cx2 = wx1 + start_x, wx1 + x
                    char_slice = orig_binary[wy1:wy2 + 1, cx1:cx2 + 1]
                    r = np.any(char_slice, axis=1)
                    if np.any(r):
                        ry1, ry2 = np.where(r)[0][[0, -1]]
                        boxes.append((int(cx1), int(wy1 + ry1), int(cx2), int(wy1 + ry2)))

        if in_char and (word_w - start_x) >= min_char_w:
            cx1, cx2 = wx1 + start_x, wx2
            char_slice = orig_binary[wy1:wy2 + 1, cx1:cx2 + 1]
            r = np.any(char_slice, axis=1)
            if np.any(r):
                ry1, ry2 = np.where(r)[0][[0, -1]]
                boxes.append((int(cx1), int(wy1 + ry1), int(cx2), int(wy1 + ry2)))

        return boxes if len(boxes) > 1 else [(int(wx1), int(wy1), int(wx2), int(wy2))]

    @staticmethod
    def _to_binary(image: Image.Image) -> np.ndarray:
        if image.mode == "RGBA":
            arr = np.array(image, dtype=np.uint8)
            # Detect dark strokes or opaque strokes
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            if gray.mean() > 127:
                return (gray < 200) & (arr[:, :, 3] > 50)
            return (gray > 50) & (arr[:, :, 3] > 50)

        img = image.convert("L")
        arr = np.array(img, dtype=np.uint8)
        if arr.mean() > 127:
            return arr < 200
        return arr > 50

    @staticmethod
    def _extract_boxes(
        labeled: np.ndarray,
        num_features: int,
        min_area: float,
        wy1: int,
        wy2: int,
        orig_binary: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        boxes = []
        word_h = max(1, wy2 - wy1 + 1)
        matra_cutoff_y = wy1 + int(word_h * 0.38)

        for label_id in range(1, num_features + 1):
            component = labeled == label_id
            if component.sum() < min_area:
                continue

            r_comp = np.any(component, axis=1)
            c_comp = np.any(component, axis=0)
            cy_min, cy_max = np.where(r_comp)[0][[0, -1]]
            cx_min, cx_max = np.where(c_comp)[0][[0, -1]]

            # Drop leftover Matra tails that sit only in the top 38% or are vertically tiny
            if cy_max <= matra_cutoff_y or (cy_max - cy_min) < (word_h * 0.25):
                continue

            full_slice = orig_binary[wy1:wy2 + 1, cx_min:cx_max + 1]
            rows = np.any(full_slice, axis=1)
            if np.any(rows):
                ry1, ry2 = np.where(rows)[0][[0, -1]]
                y_min, y_max = wy1 + ry1, wy1 + ry2
            else:
                y_min, y_max = cy_min, cy_max

            boxes.append((int(cx_min), int(y_min), int(cx_max), int(y_max)))
        return boxes

    @staticmethod
    def _merge_overlapping_boxes(
        boxes: list[tuple[int, int, int, int]],
        gap: int,
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []

        changed = True
        while changed:
            changed = False
            merged = [boxes[0]]
            for cur in boxes[1:]:
                prev = merged[-1]
                prev_w = max(1, prev[2] - prev[0])
                cur_w = max(1, cur[2] - cur[0])
                overlap = min(prev[2], cur[2]) - max(prev[0], cur[0])

                # Merge if one box horizontally overlaps most of another (e.g. dots/diacritics) or is within gap
                if (cur[0] - prev[2]) <= gap or (overlap > 0.4 * min(prev_w, cur_w)):
                    merged[-1] = (
                        min(prev[0], cur[0]),
                        min(prev[1], cur[1]),
                        max(prev[2], cur[2]),
                        max(prev[3], cur[3]),
                    )
                    changed = True
                else:
                    merged.append(cur)
            boxes = merged

        return boxes

    def _crop_character(
        self,
        rgb_arr: np.ndarray,
        box: tuple[int, int, int, int],
        img_h: int,
        img_w: int,
    ) -> Image.Image:
        x_min, y_min, x_max, y_max = box
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)
        pad = int(max(bw, bh) * self.padding_ratio)

        cx1 = max(0, x_min - pad)
        cy1 = max(0, y_min - pad)
        cx2 = min(img_w, x_max + pad + 1)
        cy2 = min(img_h, y_max + pad + 1)

        crop = rgb_arr[cy1:cy2, cx1:cx2]
        return self._make_square(crop)

    @staticmethod
    def _make_square(crop_arr: np.ndarray) -> Image.Image:
        """
        Pad crop array to a square using WHITE (255) pixels so it matches the
        white canvas background and doesn't break inversion in prediction.py.
        """
        h, w = crop_arr.shape[:2]
        side = max(h, w)
        if h == w:
            return Image.fromarray(crop_arr)

        channels = crop_arr.shape[2] if crop_arr.ndim == 3 else 1
        if channels > 1:
            square = np.full((side, side, channels), 255, dtype=crop_arr.dtype)
        else:
            square = np.full((side, side), 255, dtype=crop_arr.dtype)

        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop_arr

        return Image.fromarray(square)