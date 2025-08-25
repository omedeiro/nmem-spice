"""Waveform generation utilities."""

import numpy as np
from typing import Tuple, List, Optional
from ltspice_core.constants import (
    DEFAULT_CYCLE_TIME,
    DEFAULT_PULSE_SIGMA,
    DEFAULT_HOLD_WIDTH_WRITE,
    DEFAULT_HOLD_WIDTH_READ,
    DEFAULT_HOLD_WIDTH_CLEAR,
    DEFAULT_DT,
    DEFAULT_WRITE_AMPLITUDE,
    DEFAULT_READ_AMPLITUDE,
    DEFAULT_ENAB_WRITE_AMPLITUDE,
    DEFAULT_ENAB_READ_AMPLITUDE,
    DEFAULT_CLEAR_AMPLITUDE,
)


def flat_top_gaussian(
    t_center: float, sigma: float, hold_width: float, amp: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a flat-top Gaussian pulse.

    Args:
        t_center: Center time of the pulse
        sigma: Standard deviation of Gaussian edges
        hold_width: Width of the flat-top region
        amp: Amplitude of the pulse
        dt: Time step

    Returns:
        Tuple of (time_array, current_array)
    """
    total_width = 8 * sigma + hold_width
    t_start = t_center - total_width / 2
    t_end = t_center + total_width / 2

    t_rise = np.arange(t_start, t_center - hold_width / 2, dt)
    t_hold = np.arange(t_center - hold_width / 2, t_center + hold_width / 2, dt)
    t_fall = np.arange(t_center + hold_width / 2, t_end, dt)

    i_rise = amp * np.exp(-0.5 * ((t_rise - (t_center - hold_width / 2)) / sigma) ** 2)
    i_hold = np.full_like(t_hold, amp)
    i_fall = amp * np.exp(-0.5 * ((t_fall - (t_center + hold_width / 2)) / sigma) ** 2)

    t = np.concatenate([t_rise, t_hold[1:], t_fall[1:]])
    i = np.concatenate([i_rise, i_hold[1:], i_fall[1:]])
    return t, i


class WaveformGenerator:
    """Class for generating memory protocol waveforms."""

    def __init__(
        self,
        cycle_time: float = DEFAULT_CYCLE_TIME,
        pulse_sigma: float = DEFAULT_PULSE_SIGMA,
        hold_width_write: float = DEFAULT_HOLD_WIDTH_WRITE,
        hold_width_read: float = DEFAULT_HOLD_WIDTH_READ,
        hold_width_clear: float = DEFAULT_HOLD_WIDTH_CLEAR,
        write_amplitude: float = DEFAULT_WRITE_AMPLITUDE,
        read_amplitude: float = DEFAULT_READ_AMPLITUDE,
        enab_write_amplitude: float = DEFAULT_ENAB_WRITE_AMPLITUDE,
        enab_read_amplitude: float = DEFAULT_ENAB_READ_AMPLITUDE,
        clear_amplitude: float = DEFAULT_CLEAR_AMPLITUDE,
        dt: float = DEFAULT_DT,
    ):
        """Initialize waveform generator with default parameters."""
        self.cycle_time = cycle_time
        self.pulse_sigma = pulse_sigma
        self.hold_width_write = hold_width_write
        self.hold_width_read = hold_width_read
        self.hold_width_clear = hold_width_clear
        self.write_amplitude = write_amplitude
        self.read_amplitude = read_amplitude
        self.enab_write_amplitude = enab_write_amplitude
        self.enab_read_amplitude = enab_read_amplitude
        self.clear_amplitude = clear_amplitude
        self.dt = dt

    def get_default_patterns(self) -> List[List[str]]:
        """Get default memory test patterns."""
        return [
            [
                "null",
                "null",
                "null",
                "write_1",
                "read",
                "null",
                "null",
                "null",
                "write_0",
                "read",
            ],
            [
                "null",
                "null",
                "write_1",
                "null",
                "read",
                "null",
                "null",
                "write_0",
                "null",
                "read",
            ],
            [
                "null",
                "write_0",
                "null",
                "null",
                "read",
                "null",
                "write_1",
                "null",
                "null",
                "read",
            ],
            [
                "null",
                "write_1",
                "enab",
                "read",
                "read",
                "write_0",
                "enab",
                "read",
                "read",
                "null",
            ],
            [
                "null",
                "write_0",
                "enab",
                "read",
                "read",
                "write_1",
                "enab",
                "read",
                "read",
                "null",
            ],
        ]

    def generate_memory_protocol_sequence(
        self,
        patterns: Optional[List[List[str]]] = None,
        seed: Optional[int] = None,
        phase_offset: float = 0,
        custom_enab_disable: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Generate a complete memory protocol sequence.

        Args:
            patterns: List of operation patterns. If None, uses default patterns.
            seed: Random seed for reproducibility
            phase_offset: Phase offset for enable signal
            custom_enab_disable: Custom list of indices where enable should be disabled

        Returns:
            Tuple of (t_chan, i_chan, t_enab, i_enab, ops, enab_on)
        """
        if seed is not None:
            np.random.seed(seed)

        if patterns is None:
            patterns = self.get_default_patterns()

        # Flatten patterns into operations list
        ops = []
        for pattern in patterns:
            ops.extend(pattern)

        t_chan, i_chan = [], []
        t_enab, i_enab = [], []

        # Set up enable control
        enab_on = np.ones(len(ops), dtype=bool)
        if custom_enab_disable is None:
            # Default disable pattern
            enab_on[33:34] = False
            enab_on[37:38] = False
            enab_on[43:44] = False
            enab_on[47:48] = False
        else:
            enab_on[custom_enab_disable] = False

        hold_width_enab = 100e-9

        for i, op in enumerate(ops):
            t_center = i * self.cycle_time + self.cycle_time / 2

            # Generate data line signal (I_chan)
            if op == "write_1":
                amp = self.write_amplitude
                t_vec, i_vec = flat_top_gaussian(
                    t_center, self.pulse_sigma, self.hold_width_write, amp, self.dt
                )
                t_chan.extend(t_vec)
                i_chan.extend(i_vec)
            elif op == "write_0":
                amp = -self.write_amplitude
                t_vec, i_vec = flat_top_gaussian(
                    t_center, self.pulse_sigma, self.hold_width_write, amp, self.dt
                )
                t_chan.extend(t_vec)
                i_chan.extend(i_vec)
            elif op == "read":
                amp = self.read_amplitude
                t_vec, i_vec = flat_top_gaussian(
                    t_center, self.pulse_sigma, self.hold_width_read, amp, self.dt
                )
                t_chan.extend(t_vec)
                i_chan.extend(i_vec)
            elif op == "null":
                amp = 0
                t_vec, i_vec = flat_top_gaussian(
                    t_center, self.pulse_sigma, hold_width_enab, amp, self.dt
                )
                t_chan.extend(t_vec)
                i_chan.extend(i_vec)

            # Generate enable signal (I_enab)
            if op in ["write_1", "write_0", "enab"]:
                amp = self.enab_write_amplitude
                hold = hold_width_enab
            elif op == "read":
                amp = self.enab_read_amplitude
                hold = hold_width_enab
            elif op == "clear":
                amp = self.clear_amplitude
                hold = self.hold_width_clear
            else:
                amp = 0

            if amp > 0 and enab_on[i]:
                t_vec, i_vec = flat_top_gaussian(
                    t_center, self.pulse_sigma, hold_width_enab, amp, self.dt
                )
                t_enab.extend(t_vec + phase_offset)
                i_enab.extend(i_vec)

        return (
            np.array(t_chan),
            np.array(i_chan),
            np.array(t_enab),
            np.array(i_enab),
            ops,
            enab_on,
        )


# Convenience function for backward compatibility
def generate_memory_protocol_sequence(
    cycle_time: float = DEFAULT_CYCLE_TIME,
    pulse_sigma: float = DEFAULT_PULSE_SIGMA,
    hold_width_write: float = DEFAULT_HOLD_WIDTH_WRITE,
    hold_width_read: float = DEFAULT_HOLD_WIDTH_READ,
    hold_width_clear: float = DEFAULT_HOLD_WIDTH_CLEAR,
    write_amplitude: float = DEFAULT_WRITE_AMPLITUDE,
    read_amplitude: float = DEFAULT_READ_AMPLITUDE,
    enab_write_amplitude: float = DEFAULT_ENAB_WRITE_AMPLITUDE,
    enab_read_amplitude: float = DEFAULT_ENAB_READ_AMPLITUDE,
    clear_amplitude: float = DEFAULT_CLEAR_AMPLITUDE,
    dt: float = DEFAULT_DT,
    seed: Optional[int] = None,
    phase_offset: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Convenience function to generate memory protocol sequence with default parameters."""
    generator = WaveformGenerator(
        cycle_time=cycle_time,
        pulse_sigma=pulse_sigma,
        hold_width_write=hold_width_write,
        hold_width_read=hold_width_read,
        hold_width_clear=hold_width_clear,
        write_amplitude=write_amplitude,
        read_amplitude=read_amplitude,
        enab_write_amplitude=enab_write_amplitude,
        enab_read_amplitude=enab_read_amplitude,
        clear_amplitude=clear_amplitude,
        dt=dt,
    )
    return generator.generate_memory_protocol_sequence(
        seed=seed, phase_offset=phase_offset
    )
