from __future__ import annotations

import textwrap
import unittest

from tools.chronos_integrity_check import inspect_source


class IntegrityCheckTests(unittest.TestCase):
    def test_rejects_entity_agnostic_cv_and_holdout(self) -> None:
        source = textwrap.dedent(
            """
            from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
            cv = StratifiedKFold(n_splits=5)
            cross_validate(pipe, X, y, cv=cv)
            train_test_split(X, y)
            """
        )
        codes = {finding["code"] for finding in inspect_source(source)}
        self.assertEqual(
            codes,
            {
                "ENTITY_SPLIT_NOT_GROUP_AWARE",
                "GROUP_SPLITTER_MISSING",
                "CV_GROUPS_MISSING",
                "HOLDOUT_GROUPS_MISSING",
            },
        )

    def test_accepts_group_isolated_cv_and_holdout(self) -> None:
        source = textwrap.dedent(
            """
            from sklearn.model_selection import StratifiedGroupKFold, cross_validate
            cv = StratifiedGroupKFold(n_splits=5)
            cross_validate(pipe, X, y, cv=cv, groups=groups)
            cv.split(X, y, groups=groups)
            """
        )
        self.assertEqual(inspect_source(source), [])


if __name__ == "__main__":
    unittest.main()
