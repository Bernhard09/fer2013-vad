# src/vad_model.py
#
# Wrapper for VAD multi-task ONNX model inference.
# Input  : face crop (BGR or RGB numpy array)
# Output : emotion label (str), valence (float), arousal (float), dominance (float)

import numpy as np
import onnxruntime as ort
import cv2


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# VAD reference values per class (for display / downstream use)
VAD_REFERENCE = {
    "angry":    (-1.00,  0.75,  0.50),
    "disgust":  (-0.50,  0.25,  0.25),
    "fear":     (-0.75,  0.50, -0.75),
    "happy":    ( 0.75,  0.50,  0.25),
    "neutral":  ( 0.00,  0.00,  0.00),
    "sad":      (-0.75, -0.25, -0.50),
    "surprise": ( 0.25,  0.75, -0.25),
}

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VADModel:
    """
    Loads and runs the multi-task VAD ONNX model.

    Usage:
        model  = VADModel("models/vad_multitask.onnx")
        result = model.predict(face_crop_bgr)
        print(result)
        # {
        #   "emotion": "happy",
        #   "valence": 0.72,
        #   "arousal": 0.48,
        #   "dominance": 0.31,
        #   "confidence": 0.91
        # }
    """

    INPUT_SIZE = 224

    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Verify expected I/O
        input_names  = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        assert "face_crop"  in input_names,  f"Unexpected input names: {input_names}"
        assert "cls_logits" in output_names, f"Unexpected output names: {output_names}"
        assert "vad"        in output_names, f"Unexpected output names: {output_names}"

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Converts a BGR face crop to a normalized float32 tensor.
        Shape: (1, 3, 224, 224)
        """
        rgb   = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor     = normalized.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
        return tensor

    def predict(self, bgr_face_crop: np.ndarray) -> dict:
        """
        Runs inference on a single face crop.

        Args:
            bgr_face_crop: numpy array in BGR format (from OpenCV).
                           Any size — will be resized internally.

        Returns:
            dict with keys: emotion, valence, arousal, dominance, confidence
            Returns None if the input is invalid.
        """
        if bgr_face_crop is None or bgr_face_crop.size == 0:
            return None

        tensor  = self._preprocess(bgr_face_crop)
        outputs = self.session.run(
            ["cls_logits", "vad"],
            {"face_crop": tensor}
        )

        cls_logits = outputs[0][0]   # (7,)
        vad_values = outputs[1][0]   # (3,)

        # Softmax for confidence score
        exp_logits = np.exp(cls_logits - cls_logits.max())
        probs      = exp_logits / exp_logits.sum()

        class_idx  = int(np.argmax(probs))
        emotion    = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

        valence    = float(np.clip(vad_values[0], -1.0, 1.0))
        arousal    = float(np.clip(vad_values[1], -1.0, 1.0))
        dominance  = float(np.clip(vad_values[2], -1.0, 1.0))

        return {
            "emotion":   emotion,
            "valence":   round(valence,   3),
            "arousal":   round(arousal,   3),
            "dominance": round(dominance, 3),
            "confidence": round(confidence, 3),
        }

    def predict_batch(self, bgr_face_crops: list) -> list:
        """
        Runs inference on a list of face crops.
        Returns a list of result dicts in the same order.
        Invalid crops return None.
        """
        return [self.predict(crop) for crop in bgr_face_crops]