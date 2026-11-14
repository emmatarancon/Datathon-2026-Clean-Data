"""Tests for rag2/hybrid.py: tokenizer, BM25, fusion, example selection."""
import unittest

from rag2 import data, hybrid


def fake_embed(texts):
    from rag2 import mock
    return mock.mock_embed(texts)


class TokenizerTest(unittest.TestCase):
    def test_numbers_survive(self):
        # the BM25 tokenizer from the spec keeps decimals but splits fractions
        tokens = hybrid.tokenize("The train travels 3.5 km in 1/2 hour!")
        self.assertEqual(tokens, ["train", "travels", "3.5", "km", "1", "2", "hour"])

    def test_lowercases_and_drops_stopwords(self):
        self.assertEqual(hybrid.tokenize("The Cat AND the dog"), ["cat", "dog"])

    def test_deterministic(self):
        text = "Eleven bags with 2 apples each"
        self.assertEqual(hybrid.tokenize(text), hybrid.tokenize(text))


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.docs = [
            ("d0", "Anna has 3 bags with 4 apples and eats 2 apples."),
            ("d1", "A train travels 12 km in 20 minutes. How fast?"),
            ("d2", "What is the value of 2 + 3 * 4?"),
            ("d3", "Anna has 3 bags with 5 oranges."),
        ]
        self.bm25 = hybrid.BM25(self.docs)

    def test_exact_numbers_rank_first(self):
        scores = self.bm25.score("3 bags with 4 apples")
        best = max(scores, key=scores.get)
        self.assertIn(self.docs[best][0], ("d0", "d3"))  # both share "3 bags"
        self.assertGreater(scores.get(self.bm25.ids.index("d0"), 0.0),
                           scores.get(self.bm25.ids.index("d1"), 0.0))

    def test_deterministic_scores(self):
        query = "train 12 km"
        self.assertEqual(self.bm25.score(query), self.bm25.score(query))

    def test_unknown_terms_are_safe(self):
        self.assertEqual(self.bm25.score("zzz qqq 42"), {})

    def test_payload_roundtrip(self):
        payload = self.bm25.payload("fp")
        clone = hybrid.BM25.from_payload(payload)
        query = "3 bags with 4 apples"
        self.assertEqual(self.bm25.score(query), clone.score(query))


class FusionTest(unittest.TestCase):
    def test_minmax_bounds(self):
        normalized = hybrid.minmax([(0, 1.0), (1, 2.0), (2, 1.5)])
        self.assertAlmostEqual(min(normalized.values()), 0.0)
        self.assertAlmostEqual(max(normalized.values()), 1.0)

    def test_minmax_degenerate(self):
        self.assertEqual(hybrid.minmax([(0, 2.0), (1, 2.0)]), {0: 0.0, 1: 0.0})

    def test_w_one_is_cosine_only(self):
        cos = [(0, 0.9), (1, 0.5), (2, 0.1)]
        bm = [(2, 9.0), (1, 3.0)]
        fused = hybrid.fuse(cos, bm, w=1.0)
        self.assertEqual([idx for idx, _s in fused], [0, 1, 2])

    def test_fusion_mixes_both_lists(self):
        cos = [(0, 0.9), (1, 0.1)]
        bm = [(1, 5.0), (0, 1.0)]
        for w in (0.85, 0.7, 0.5):
            fused = hybrid.fuse(cos, bm, w=w)
            self.assertEqual({idx for idx, _s in fused}, {0, 1})
        self.assertEqual(hybrid.fuse(cos, bm, w=0.0)[0][0], 1)  # bm25 dominates
        self.assertEqual(hybrid.fuse(cos, bm, w=1.0)[0][0], 0)  # cosine dominates

    def test_rrf(self):
        # the top-1 of both lists beats every other candidate
        cos = [(1, 0.9), (0, 0.2)]
        bm = [(1, 5.0), (0, 1.0)]
        fused = hybrid.rrf(cos, bm)
        self.assertEqual([idx for idx, _s in fused], [1, 0])


class SelectExamplesTest(unittest.TestCase):
    def setUp(self):
        self._old = data._MOCK_EMBED
        data._MOCK_EMBED = fake_embed
        problems = [
            {"id": "a", "language": "eng", "text": "Anna has 3 bags with 4 apples. How many apples does she have left?", "options": {"A": "8", "B": "9"}, "label": "B"},
            {"id": "b", "language": "eng", "text": "Anna has 3 bags with 4 apples each. She eats 2 apples.", "options": {"A": "10", "B": "12"}, "label": "A"},
            {"id": "c", "language": "eng", "text": "A square has a perimeter of 36 cm. What is its area?", "options": {"A": "81 cm2", "B": "144 cm2"}, "label": "A"},
        ]
        self.index = data.MemIndex.from_problems(problems, fake_embed(
            [hybrid.doc_text(p) for p in problems]))
        self.bm25, self.elapsed, self.cached = hybrid.build_bm25(
            self.index, cache_path=None)
        self.problem = {"id": "q", "language": "eng", "text": "Anna has 3 bags with 4 apples and eats 1. How many are left?",
                        "options": {"A": "11", "B": "12"}}

    def tearDown(self):
        data._MOCK_EMBED = self._old

    def test_build_bm25_no_cache(self):
        self.assertFalse(self.cached)
        self.assertEqual(self.bm25.n, 3)

    def test_selection_shape_and_order(self):
        examples, info = hybrid.select_examples(self.problem, self.index, self.bm25)
        self.assertLessEqual(len(examples), 2)
        self.assertTrue(all(e.get("options") for e in examples))
        self.assertLessEqual(info["estimated_seconds"], 35.0)
        self.assertIn("a", info["cos_top1"] + info["bm25_top1"])

    def test_w_one_matches_cosine_top1(self):
        _ex, info = hybrid.select_examples(self.problem, self.index, self.bm25, w=1.0)
        _ex2, info_cos = hybrid.select_examples(self.problem, self.index, self.bm25,
                                                w=0.0)
        # with w=0 bm25 alone decides; both top1 ids exist and are reported
        self.assertTrue(info["fused_top1"])
        self.assertTrue(info_cos["fused_top1"])

    def test_top1_related(self):
        problem = {"id": "q", "level": 2, "year": 2020, "text": "3 bags and 4 apples"}
        neighbor_same = {"level": 2, "year": 2020, "text": "3 apples and 4 bags"}
        self.assertTrue(hybrid.top1_related(problem, neighbor_same)["related"])
        neighbor_numbers = {"level": 4, "year": 2011, "text": "4 bags and 3 apples"}
        self.assertTrue(hybrid.top1_related(problem, neighbor_numbers)["related"])
        neighbor_other = {"level": 4, "year": 2011, "text": "17 trains and 2 stations"}
        self.assertFalse(hybrid.top1_related(problem, neighbor_other)["related"])


if __name__ == "__main__":
    unittest.main()
