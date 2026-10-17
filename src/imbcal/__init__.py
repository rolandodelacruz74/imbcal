"""imbcal: calibrated probability estimation under class imbalance.

Main entry point: :class:`ImbalanceCalibratedClassifier`. The safe default is to fit at the natural prevalence and
recalibrate (``strategy="none"``); prevalence-preserving augmentation (``strategy="augment"``) can use any synthetic
data generator and is selected by cross-validation of the recalibrated loss, with no augmentation always available.
"""
from .calibration import IsotonicCalibrator, PlattCalibrator, crossfit_calibrated
from .diagnostics import majority_exposure
from .estimator import Candidate, ImbalanceCalibratedClassifier
from .generators import BaseGenerator, CallableGenerator, GaussianGenerator, SMOTEGenerator
from .metrics import (calibration_slope_intercept, expected_calibration_error, paired_comparison,
                      probability_report)

__version__ = "0.1.0.dev0"
__all__ = ["ImbalanceCalibratedClassifier", "Candidate", "BaseGenerator", "SMOTEGenerator", "GaussianGenerator",
           "CallableGenerator", "PlattCalibrator", "IsotonicCalibrator", "crossfit_calibrated",
           "calibration_slope_intercept", "expected_calibration_error", "probability_report", "paired_comparison",
           "majority_exposure"]
