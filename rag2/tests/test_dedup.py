"""Tests for rag2/dedup.py: the near-duplicate answer transfer."""
import unittest

from rag2 import data, dedup

Q_VEC = [1.0]
LOW_VEC = [0.5]  # cosine 0.5 with the query


def make_index(neighbor, vec=Q_VEC):
    return data.MemIndex.from_problems([neighbor], [vec])


CURRENT = {
    "id": "val-9", "language": "eng", "level": 2, "year": 2024,
    "text": "13. Tom has 3 bags with 4 apples in each bag. He eats 2 apples. How many apples does he have left?",
    "options": {"A": "12", "B": "10", "C": "8", "D": "9", "E": "14"},
    # NOTE: no "label" key at all — transfer must never read it
}

TWIN = {
    "id": "train-7", "language": "eng", "level": 1, "year": 2019,
    "text": "7. Tom has 3 bags with 4 apples in each bag. He eats 2 apples. How many apples does he have left?",
    "options": {"A": "12", "B": "9", "C": "10", "D": "8", "E": "14"},
    "label": "C",
}


class DedupTest(unittest.TestCase):
    def setUp(self):
        self._old = data._MOCK_EMBED
        data._MOCK_EMBED = lambda texts: [Q_VEC for _ in texts]

    def tearDown(self):
        data._MOCK_EMBED = self._old

    def test_transfers_by_value_not_letter(self):
        # twin's correct value is "10" (label C); current has "10" at letter B
        letter, info = dedup.try_transfer(CURRENT, make_index(TWIN))
        self.assertEqual(letter, "B")
        self.assertEqual(info["reason"], "transferred")
        self.assertEqual(info["neighbor_id"], "train-7")

    def test_never_reads_the_current_label(self):
        problem = dict(CURRENT, label="E")  # wrong on purpose: it must not matter
        letter, _info = dedup.try_transfer(problem, make_index(TWIN))
        self.assertEqual(letter, "B")

    def test_no_transfer_when_a_number_changes(self):
        twin = dict(TWIN, text=TWIN["text"].replace("4 apples", "5 apples"))
        letter, info = dedup.try_transfer(CURRENT, make_index(twin))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "statement numbers differ")

    def test_no_transfer_when_text_is_different(self):
        twin = dict(TWIN, text="7. A train travels 12 km and then 5 km. How far in total?")
        letter, info = dedup.try_transfer(CURRENT, make_index(twin))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "text ratio below 0.90")

    def test_no_transfer_when_id_is_the_same(self):
        twin = dict(TWIN, id=CURRENT["id"])
        letter, info = dedup.try_transfer(CURRENT, make_index(twin))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "same id")

    def test_exclude_id_removes_the_only_candidate(self):
        letter, info = dedup.try_transfer(CURRENT, make_index(TWIN),
                                          exclude_id=CURRENT["id"])
        # the twin has another id, so it stays: the exclusion applies to the query id
        self.assertEqual(letter, "B")
        letter2, _info2 = dedup.try_transfer(CURRENT, make_index(TWIN),
                                             exclude_id=TWIN["id"])
        self.assertIsNone(letter2)

    def test_no_transfer_when_two_options_match_the_value(self):
        # multiset of option values is equal on both sides ("8" is duplicated
        # in both), but the wanted value "8" appears at TWO letters here
        twin = dict(TWIN, options={"A": "8", "B": "8", "C": "10", "D": "9", "E": "14"},
                    label="A")
        problem = dict(CURRENT, options={"A": "10", "B": "9", "C": "14", "D": "8", "E": "8"})
        letter, info = dedup.try_transfer(problem, make_index(twin))
        self.assertIsNone(letter)
        self.assertIn("options match", info["reason"])

    def test_no_transfer_when_option_values_differ(self):
        twin = dict(TWIN, options={"A": "12", "B": "9", "C": "11", "D": "8", "E": "14"})
        letter, info = dedup.try_transfer(CURRENT, make_index(twin))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "option values differ")

    def test_no_transfer_below_cosine_threshold(self):
        letter, info = dedup.try_transfer(CURRENT, make_index(TWIN, vec=LOW_VEC),
                                          tau_dup=0.97)
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "cosine below tau_dup")

    def test_no_transfer_without_neighbor_options(self):
        twin = dict(TWIN, options={})
        letter, info = dedup.try_transfer(CURRENT, make_index(twin))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "neighbor without read options")

    def test_no_transfer_without_current_options(self):
        problem = dict(CURRENT, options={})
        letter, info = dedup.try_transfer(problem, make_index(TWIN))
        self.assertIsNone(letter)
        self.assertEqual(info["reason"], "current problem has no read options")

    def test_fraction_forms_transfer(self):
        # twin's correct value "1/2", current writes it as "0,5": same canonical
        twin = dict(TWIN, text="7. Half of the cake is left. How much is that?",
                    options={"A": "1/2", "B": "1/3", "C": "2/3", "D": "3/4", "E": "1"},
                    label="A")
        problem = {"id": "val-10", "language": "eng",
                   "text": "Half of the cake is left. How much is that?",
                   "options": {"A": "1/3", "B": "0,5", "C": "2/3", "D": "3/4", "E": "1"}}
        letter, _info = dedup.try_transfer(problem, make_index(twin))
        self.assertEqual(letter, "B")


if __name__ == "__main__":
    unittest.main()
