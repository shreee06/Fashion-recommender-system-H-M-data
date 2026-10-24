"""Evaluate the FINAL chosen models on the TEST week (last 7 days of the data).

SKELETON. Do not run until all tuning on the validation week is finished: every
look at the test week spends it. Refuses to run without --i-am-done-tuning.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-am-done-tuning", action="store_true",
                    help="confirm that model selection and tuning are finished")
    args = ap.parse_args()
    if not args.i_am_done_tuning:
        sys.exit("Refusing to evaluate on the test week: pass --i-am-done-tuning once tuning is finished.")

    # TODO (when the final models exist):
    #  1. load transactions (fashion_recsys.data.load_transaction_codes), derive the split
    #     (fashion_recsys.config.make_split) and the same seeded customer sample;
    #  2. build_eval_set(..., test_start, test_end) with cutoff = test_start;
    #  3. fit/recommend with the final chosen models, evaluate_cold_warm, and write
    #     evals/results/final_test.json exactly once.
    raise NotImplementedError("no final models registered yet")


if __name__ == "__main__":
    main()
