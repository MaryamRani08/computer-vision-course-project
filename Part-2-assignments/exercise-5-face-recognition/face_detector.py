import cv2
import numpy as np
from mtcnn import MTCNN


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        self.tm_window_size = tm_window_size
        self.tm_threshold = tm_threshold
         

    # EX5.1 a) Tracking Faces in Videos
    # TODO: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            self.reference = self.detect_face(image)
            return self.reference

        # Reference template and bounding box
        ref_img= self.reference["image"]
        ref_box = self.reference["rect"]
        x, y, w, h = ref_box
        template = self.crop_face(ref_img,ref_box)

        # Search window in current frame.
        h_img, w_img = image.shape[:2]
        dx, dy = self.tm_window_size, self.tm_window_size
        x1 = max(x - dx, 0) 
        y1 = max(y - dy, 0)
        x2 = min(x + w + dx, w_img) 
        y2 = min(y + h + dy, h_img) 
        search_window = image[y1:y2, x1:x2]

        # using normalized cross correlation to match template
        result = cv2.matchTemplate(search_window, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if min_val < self.tm_threshold:
            detection = self.detect_face(image)
            if detection:
                self.reference = detection
            return detection

        # EX5.1 b) Alignment for Pose Normalization
        top_left = (x1 + min_loc[0], y1 + min_loc[1])
        new_box = [top_left[0], top_left[1], w, h]
        aligned_face = self.align_face(image, new_box)
        new_ref = {"rect": new_box, "image": image, "aligned": aligned_face, "response": min_val}
        self.reference = new_ref
        return new_ref

    def detect_face(self, image):
        if not (
            detections := self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
        ):
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )
    # Cropping face
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]