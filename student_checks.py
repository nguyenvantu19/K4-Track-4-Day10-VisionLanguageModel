"""Small CPU checks for the student's coordinate/output contract."""
import json
import unittest

from scripts.score import evaluate
from student_lab import objects_to_answer, yolo_to_box


class CoordinateChecks(unittest.TestCase):
    def test_normalized_box_roundtrip_matches_official_scorer(self):
        row = {"image_id": "check", "width": 640, "height": 360,
               "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
        raw = objects_to_answer(row["objects"])
        result = evaluate([row], {"check": raw}, n_boot=20)
        self.assertEqual(result["micro"]["f1"], 1.0)
        wrong = evaluate([row], {"check": raw}, coord="pixel", n_boot=20)
        self.assertEqual(wrong["micro"]["f1"], 0.0)

    def test_empty_output_and_boundary_clipping(self):
        self.assertEqual(objects_to_answer([]), "[]")
        result = yolo_to_box("crack", [0, 1, 0.2, 0.2])
        self.assertEqual(result["bbox_2d"], [0, 900, 100, 1000])

    def test_annotations_have_stable_spatial_order(self):
        objects = [{"label": "manhole", "yolo": [0.8, 0.8, 0.1, 0.1]},
                   {"label": "crack", "yolo": [0.2, 0.2, 0.1, 0.1]}]
        self.assertEqual(objects_to_answer(objects), objects_to_answer(list(reversed(objects))))
        self.assertEqual(json.loads(objects_to_answer(objects))[0]["label"], "crack")


if __name__ == "__main__":
    unittest.main()
