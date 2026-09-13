"""Reject the incomplete evidence that previously escaped the PR #302 harness."""
import copy
import unittest
from bench_pr302_abba import ARMS, CTX_SIZE, RUNGS, validate_report


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.report = {"run": {"mode": "bench-only"}, "bench": {"contextScaling": [
            {"targetTokens": n, "inputTokens": n + 10, "decodeTokPerSec": 40,
             "ttftMs": 1000, "prefillTokPerSec": 700, "runs": 1, "note": None}
            for n in RUNGS]}}
        self.log = ("[nax-sdpa] engaged: stock fused sdpa\n"
                    "[mtp-trace] rounds=32 m_avg=8.00 acc_avg=2.00\n"
                    "[spec-stats] mode=mtp\n")
        self.split = "[sdpa-split] engaged: qL=9 kL=32777 Hq=24 Hkv=4\n"

    def test_balanced_complete_abba_blocks_and_context_headroom(self):
        self.assertEqual("".join(ARMS), "ABBAABBA")
        self.assertEqual(ARMS.count("A"), ARMS.count("B"))
        self.assertGreaterEqual(ARMS.count("A"), 3)
        self.assertGreater(CTX_SIZE, 32890 + 192)

    def test_both_valid_arms(self):
        validate_report(self.report, self.log + self.split, "A")
        validate_report(self.report, self.log, "B")

    def test_reject_old_4k_substitution(self):
        self.report["bench"]["contextScaling"][0]["targetTokens"] = 4096
        with self.assertRaisesRegex(ValueError, "incorrect context"):
            validate_report(self.report, self.log, "B")

    def test_reject_32k_http400_or_absent_rung(self):
        bad = copy.deepcopy(self.report)
        bad["bench"]["contextScaling"][-1].update(runs=0, inputTokens=None, note="HTTP 400")
        with self.assertRaisesRegex(ValueError, "failed context"):
            validate_report(bad, self.log, "B")
        self.report["bench"]["contextScaling"].pop()
        with self.assertRaisesRegex(ValueError, "incorrect context"):
            validate_report(self.report, self.log, "B")

    def test_reject_wrong_or_missing_engagement(self):
        for log, arm in [(self.log, "A"), (self.log + self.split, "B"),
                         (self.log.replace("m_avg=8.00", "m_avg=6.00"), "B")]:
            with self.assertRaises(ValueError):
                validate_report(self.report, log, arm)


if __name__ == "__main__":
    unittest.main()
