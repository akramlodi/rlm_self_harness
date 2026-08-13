"""The experiment package's one error base.

Every failure the scaffold raises on its own behalf says the same thing at
bottom: the persisted experiment state contradicts the configuration (or
itself), so continuing would silently mix two experiments. Each stage keeps
its own specific class -- ``SplitsPersistenceError``,
``ExperimentPersistenceError`` and its ``IdentityMismatchError``,
``MiningBudgetExceededError``, ``EvaluationPersistenceError``,
``ReportInputError`` -- because the message and the recovery differ; they all
derive from ``ExperimentError`` so a caller driving the whole pipeline can
catch the family in one clause instead of enumerating five siblings.

``ExperimentError`` subclasses ``RuntimeError``, which every one of those
classes already did, so existing ``except RuntimeError`` handlers keep working
unchanged.
"""


class ExperimentError(RuntimeError):
    """Base for every error the experiment scaffold raises on its own behalf."""


__all__ = ["ExperimentError"]
