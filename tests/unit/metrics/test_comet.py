import unittest

from telescope.metrics.comet.metric import COMET


class TestCOMET(unittest.TestCase):
    
    comet = COMET()

    def test_score(self): 
        # README example!
        src = [
            "Dem Feuer konnte Einhalt geboten werden",
            "Schulen und Kindergärten wurden eröffnet."
        ] 
        cand = [
            "The fire could be stopped",
            "Schools and kindergartens were open"
        ]
        ref = [
            "They were able to control the fire.",
            "Schools and kindergartens opened"
        ]
        expected_seg = [0.19016997516155243, 0.9156627655029297]
        expected_sys = 0.5529163703322411
        result = self.comet.score(src, cand, ref)
        
        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand) 
    
    def test_name_property(self):
        self.assertEqual(self.comet.name, "COMET")