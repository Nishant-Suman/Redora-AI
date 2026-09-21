"""
Image Preprocessing Pipeline for Document & Scanned PDF OCR Enhancement.
Includes Deskewing, Contrast Enhancement, Denoising, Border Removal, and Binarization.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
import cv2
from PIL import Image


class ImagePreprocessor:
    """Advanced Computer Vision preprocessor for document & scanned PDF images."""

    def __init__(self):
        pass

    @staticmethod
    def to_cv2_image(image_input: Union[Image.Image, np.ndarray, bytes, str]) -> np.ndarray:
        """Converts various image input types to an OpenCV BGR numpy array."""
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            return img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def to_pil_image(cv2_img: np.ndarray) -> Image.Image:
        """Converts an OpenCV image (BGR or Grayscale) to a PIL Image (RGB)."""
        if len(cv2_img.shape) == 2:
            return Image.fromarray(cv2_img).convert("RGB")
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def detect_skew_angle(self, cv2_img: np.ndarray) -> float:
        """
        Detects document skew angle using edge detection and Hough transform / minAreaRect.
        Returns angle in degrees (positive or negative).
        """
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) if len(cv2_img.shape) == 3 else cv2_img
        
        # Invert and blur
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Use morphological operations to link text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of text lines
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            # OpenCV minAreaRect returns angle in [-90, 0)
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle to resist outliers
        median_angle = float(np.median(angles))
        return round(median_angle, 2)

    def deskew(self, cv2_img: np.ndarray, angle: float = None) -> Tuple[np.ndarray, float]:
        """
        Rotates image to correct skew angle.
        If angle is not provided, it is automatically calculated.
        """
        if angle is None:
            angle = self.detect_skew_angle(cv2_img)

        if abs(angle) < 0.2:
            return cv2_img, 0.0

        h, w = cv2_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions to prevent clipping
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        # Rotate with white background
        rotated = cv2.warpAffine(
            cv2_img,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated, angle

    def denoise(self, cv2_img: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Denoises document image while preserving sharp character edges.
        Methods: 'bilateral', 'fast_nl_means', 'median'
        """
        if method == "bilateral":
            # Bilateral filter smoothens flat regions while keeping text edges sharp
            if len(cv2_img.shape) == 3:
                return cv2.bilateralFilter(cv2_img, d=9, sigmaColor=75, sigmaSpace=75)
            else:
                return cv2.bilateralFilter(cv2_img, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == "fast_nl_means":
            if len(cv2_img.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(cv2_img, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(cv2_img, None, 10, 7, 21)
        elif method == "median":
            return cv2.medianBlur(cv2_img, 3)
        return cv2_img

    def enhance_contrast(self, cv2_img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to normalize uneven lighting, shadows, and faint ink.
        """
        if len(cv2_img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(cv2_img)

    def binarize(self, cv2_img: np.ndarray, method: str = "otsu") -> np.ndarray:
        """
        Binarizes image into crisp black and white for optimal OCR recognition.
        Methods: 'otsu', 'adaptive_gaussian', 'adaptive_mean'
        """
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) if len(cv2_img.shape) == 3 else cv2_img
        
        if method == "otsu":
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "adaptive_gaussian":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "adaptive_mean":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        return gray

    def remove_borders(self, cv2_img: np.ndarray) -> np.ndarray:
        """Removes scanner dark margins and page boundary shadows."""
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) if len(cv2_img.shape) == 3 else cv2_img
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2_img
        
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img_h, img_w = cv2_img.shape[:2]
        
        # Only crop if bounding box occupies a substantial portion
        if w > img_w * 0.4 and h > img_h * 0.4:
            # Add small safety margin
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_w, x + w + margin)
            y2 = min(img_h, y + h + margin)
            return cv2_img[y1:y2, x1:x2]
        return cv2_img

    def preprocess_pipeline(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes, str],
        do_deskew: bool = True,
        do_denoise: bool = True,
        do_enhance: bool = True,
        do_binarize: bool = True,
        binarize_method: str = "otsu",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes complete computer vision enhancement pipeline.
        Returns PIL Image and metadata dictionary.
        """
        cv_img = self.to_cv2_image(image_input)
        meta = {"original_shape": cv_img.shape[:2]}

        # 1. Deskew
        if do_deskew:
            cv_img, detected_angle = self.deskew(cv_img)
            meta["skew_angle"] = detected_angle
        else:
            meta["skew_angle"] = 0.0

        # 2. Denoise
        if do_denoise:
            cv_img = self.denoise(cv_img, method="bilateral")
            meta["denoised"] = True

        # 3. Contrast enhancement
        if do_enhance:
            cv_img = self.enhance_contrast(cv_img)
            meta["enhanced_contrast"] = True

        # 4. Binarize
        if do_binarize:
            cv_img = self.binarize(cv_img, method=binarize_method)
            meta["binarized"] = True
            meta["binarize_method"] = binarize_method

        meta["final_shape"] = cv_img.shape[:2]
        pil_result = self.to_pil_image(cv_img)
        return pil_result, meta
