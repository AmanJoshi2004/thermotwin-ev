"""Column names, split conventions and defaults shared across the project."""

from __future__ import annotations

# Measured signals a production drive controller already has.
INPUTS: list[str] = ["ambient", "coolant", "u_d", "u_q", "motor_speed", "i_d", "i_q"]

# Temperatures we want to estimate (all in degrees Celsius).
TARGETS: list[str] = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]

# Inputs that are temperatures (the thermal neural network treats them as boundary nodes).
TEMP_INPUTS: list[str] = ["ambient", "coolant"]

# Columns that exist in the Kaggle file but must never be used as model inputs.
EXCLUDED: list[str] = ["torque"]

GROUP: str = "profile_id"

# The Kaggle data is sampled at 2 Hz.
SAMPLE_HZ: float = 2.0
DT_S: float = 1.0 / SAMPLE_HZ

# Test sessions used in the literature. Use them only to compare with published numbers;
# they are development data for this course project, not the official test set.
LITERATURE_TEST_PROFILES_TNN: list[int] = [60, 62, 74]
LITERATURE_TEST_PROFILES_2021: list[int] = [65, 72]
LITERATURE_VAL_PROFILES_2021: list[int] = [58]

# Moving-average spans in samples (2 Hz): 2 min, 10 min, 30 min, 60 min.
EWMA_SPANS: list[int] = [240, 1200, 3600, 7200]

# Signals that also get an exponentially weighted standard deviation.
EWSTD_COLS: list[str] = ["i_s", "motor_speed", "u_s"]

# Demo temperature limits in degrees Celsius. The prototype motor's real limits are not
# published, so these are illustrative values inside the data's range. Change them freely.
DEFAULT_LIMITS: dict[str, float] = {
    "pm": 110.0,
    "stator_yoke": 100.0,
    "stator_tooth": 110.0,
    "stator_winding": 125.0,
}
