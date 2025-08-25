"""Constants used throughout the SPICE simulation package."""

# Plotting constants
FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3

# Default simulation parameters
DEFAULT_CYCLE_TIME = 1e-6
DEFAULT_PULSE_SIGMA = 35e-9
DEFAULT_HOLD_WIDTH_WRITE = 120e-9
DEFAULT_HOLD_WIDTH_READ = 300e-9
DEFAULT_HOLD_WIDTH_CLEAR = 5e-9
DEFAULT_DT = 0.1e-9

# Default current amplitudes (A)
DEFAULT_WRITE_AMPLITUDE = 80e-6
DEFAULT_READ_AMPLITUDE = 725e-6
DEFAULT_ENAB_WRITE_AMPLITUDE = 465e-6
DEFAULT_ENAB_READ_AMPLITUDE = 300e-6
DEFAULT_CLEAR_AMPLITUDE = 700e-6

# File extensions
SPICE_NETLIST_EXT = ".cir"
SPICE_RAW_EXT = ".raw"
PWL_EXT = ".txt"

# SPICE simulation paths (configurable)
DEFAULT_LTSPICE_PATH = (
    "/mnt/c/Users/omedeiro/AppData/Local/Programs/ADI/LTspice/LTspice.exe"
)
