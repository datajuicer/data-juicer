import unittest
import os
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from data_juicer.ops.mapper.video_calibration_mapper import VideoCalibrationMapper
from data_juicer.utils.constant import Fields

class TestVideoCalibrationMapper(unittest.TestCase):
    
    def setUp(self):
        self.video_path = 'test_video.mp4'
        # Create a dummy video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def test_process_single(self):
        # Mock Droid import inside process_single
        # Since Droid is imported inside process_single, we need to patch sys.modules or use patch.dict
        # But a simpler way is to mock the class where it is used.
        # However, since the import happens inside the function, standard patch object might be tricky if the module isn't loaded.
        
        # We will mock the entire process_single logic's dependency on Droid.
        # Actually, we can patch 'data_juicer.ops.mapper.video_calibration_mapper.sys.path.append' to avoid side effects
        # and patch 'builtins.__import__' or similar? No that's too complex.
        
        # Let's assume Droid is available for the test logic flow, or mock the import.
        # We can use patch.dict(sys.modules, {'droid': MagicMock()})
        
        mock_droid_module = MagicMock()
        mock_droid_class = MagicMock()
        mock_droid_module.Droid = mock_droid_class
        
        mock_instance = mock_droid_class.return_value
        mock_instance.terminate.return_value = (None, np.array([100.0, 100.0, 320.0, 240.0]))

        with patch.dict('sys.modules', {'droid': mock_droid_module}):
            # We also need to mock os.path.exists to return True for DroidCalib path check in __init__
            # and mock subprocess.run to avoid actual git clone
            with patch('os.path.exists') as mock_exists, \
                 patch('subprocess.run') as mock_run:
                
                # Make sure video path exists
                def side_effect(path):
                    if path == self.video_path:
                        return True
                    if "DroidCalib" in path:
                        return True # Pretend DroidCalib exists
                    return False
                mock_exists.side_effect = side_effect
                
                op = VideoCalibrationMapper(weights_path='dummy.pth')
                sample = {'videos': [self.video_path]}
                
                # Run
                res = op.process_single(sample)
                
                # Check
                self.assertIn(Fields.meta, res)
                self.assertIn('camera_intrinsics', res[Fields.meta])
                self.assertEqual(len(res[Fields.meta]['camera_intrinsics']), 1)
                self.assertEqual(len(res[Fields.meta]['camera_intrinsics'][0]), 4)
                
                # Verify Droid was called
                self.assertTrue(mock_droid_class.called)
                self.assertTrue(mock_instance.track.called)
                self.assertTrue(mock_instance.terminate.called)

if __name__ == '__main__':
    unittest.main()
