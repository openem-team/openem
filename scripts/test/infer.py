""" Test routines for infer.py """

import os
import subprocess
import tempfile
import unittest

import pandas as pd

class TestInfer(unittest.TestCase):
    def test_video_mode(self):
        temp_file, temp_path = tempfile.mkstemp(".csv")
        os.close(temp_file)
        cmd = ["python3",
               "/scripts/infer.py",
               "--graph-pb", "/working/deploy/detect/detect_retinanet.pb",
               "--img-base-dir", "/working/deploy/video",
               "--img-min-side", "360",
               "--img-max-side", "720",
               "--keep-threshold","0.40",
               "--csv-flavor", "video",
               "--output-csv", temp_path,
               "/working/deploy/video/work.csv"]
        proc = subprocess.Popen(cmd)
        proc.wait()

        # Verify successful execution
        self.assertEqual(proc.returncode, 0)

        expected_df = pd.read_csv("/working/deploy/video/expected.csv")
        results_df = pd.read_csv(temp_path)

        self.assertAlmostEqual(len(expected_df),
                               len(results_df),
                               delta=round(len(results_df)*0.05))

        os.unlink(temp_path)
