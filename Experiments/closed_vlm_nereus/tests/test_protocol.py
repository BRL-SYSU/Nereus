from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from Experiments.closed_vlm_nereus.apiyi_client import build_payload
from Experiments.closed_vlm_nereus.protocol import (
    COUNT_PROMPT,
    GRID_PROMPT,
    image_to_data_uri,
    parse_count,
    parse_grid,
)


class ProtocolTests(unittest.TestCase):
    def test_frozen_prompts_match_track_a_asset(self) -> None:
        self.assertEqual(
            COUNT_PROMPT,
            "How many fish are visible in this underwater image? "
            "Answer with only the total count.",
        )
        self.assertEqual(
            GRID_PROMPT,
            "Estimate the number of fish in each 3x3 grid region and "
            "identify the densest region. Return JSON.",
        )

    def test_count_normalization(self) -> None:
        self.assertEqual(parse_count("86")["count"], 86)
        self.assertEqual(parse_count('{"total_count": 12}')["count"], 12)
        self.assertEqual(parse_count("The total fish count is 9.")["count"], 9)
        self.assertFalse(parse_count("many")["valid"])

    def test_grid_normalization_and_explicit_precedence(self) -> None:
        response = """
        ```json
        {
          "grid_counts": {
            "top_left": 2, "top_center": 0, "top_right": 0,
            "middle_left": 0, "center": 1, "middle_right": 0,
            "bottom_left": 0, "bottom_center": 0, "bottom_right": 0
          },
          "densest_grid_region": "center"
        }
        ```
        """
        parsed = parse_grid(response)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["densest"], "center")
        self.assertFalse(parsed["explicit_derived_consistent"])

    def test_grid_can_derive_densest(self) -> None:
        response = """
        {"counts":{"top-left":0,"top-center":0,"top-right":0,
        "center-left":1,"center":4,"center-right":0,
        "bottom-left":0,"bottom-center":0,"bottom-right":0}}
        """
        parsed = parse_grid(response)
        self.assertEqual(parsed["densest"], "center")
        self.assertEqual(parsed["parse_method"], "derived")

    def test_payload_uses_gpt5_parameter_contract(self) -> None:
        gpt = build_payload(
            model="gpt-5",
            prompt="p",
            image_data_uri="data:image/jpeg;base64,x",
            task="grid",
        )
        gemini = build_payload(
            model="gemini-3.1-pro-preview",
            prompt="p",
            image_data_uri="data:image/jpeg;base64,x",
            task="grid",
        )
        self.assertIn("max_completion_tokens", gpt)
        self.assertNotIn("max_tokens", gpt)
        self.assertIn("max_tokens", gemini)
        self.assertNotIn("temperature", gpt)
        self.assertNotIn("top_p", gpt)

    def test_image_mime_uses_file_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "misleading.png"
            data = b"\xff\xd8\xff" + b"jpeg"
            path.write_bytes(data)
            uri = image_to_data_uri(path)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), data)


if __name__ == "__main__":
    unittest.main()
