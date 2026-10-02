import unittest

from sourcemark.locate import locate

DOC = """alpha line one
beta line two has a distinctive phrase about lighthouses
gamma line three
delta line four repeats: shared text
epsilon five
zeta six repeats: shared text
"""


class LocateTest(unittest.TestCase):
    def test_position_hit(self):
        q = "distinctive phrase about lighthouses"
        s = DOC.index(q)
        m = locate(DOC, q, hint_start=s)
        self.assertEqual((m.start, m.method, m.similarity), (s, "position", 1.0))

    def test_shifted_exact(self):
        q = "distinctive phrase about lighthouses"
        s = DOC.index(q)
        shifted = "new header line\n" * 5 + DOC
        m = locate(shifted, q, hint_start=s)
        self.assertEqual(m.method, "exact")
        self.assertEqual(shifted[m.start : m.end], q)

    def test_duplicate_disambiguated_by_context(self):
        q = "shared text"
        second = DOC.rindex(q)
        prefix = DOC[second - 20 : second]
        suffix = DOC[second + len(q) : second + len(q) + 5]
        m = locate("x\n" + DOC, q, prefix, suffix, hint_start=None)
        self.assertEqual(m.start, second + 2)

    def test_fuzzy_after_edit(self):
        q = "beta line two has a distinctive phrase about lighthouses"
        edited = DOC.replace("distinctive phrase", "very distinctive phrase")
        m = locate(edited, q)
        self.assertEqual(m.method, "fuzzy")
        self.assertGreater(m.similarity, 0.9)
        self.assertIn("lighthouses", edited[m.start : m.end])

    def test_whitespace_reflow_is_exactish(self):
        q = "beta line two has a distinctive phrase"
        reflowed = DOC.replace("has a distinctive", "has a\n    distinctive")
        m = locate(reflowed, q)
        self.assertEqual(m.method, "loose")
        self.assertEqual(m.similarity, 1.0)

    def test_absent_returns_none(self):
        self.assertIsNone(locate(DOC, "completely unrelated sentence about volcanoes erupting"))

    def test_crlf_document(self):
        q = "gamma line three"
        m = locate(DOC.replace("\n", "\r\n"), q)
        self.assertIsNotNone(m)


if __name__ == "__main__":
    unittest.main()
