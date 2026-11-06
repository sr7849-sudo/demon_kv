from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pynvml


@dataclass
class TelemetryResult:
    n_samples: int
    sample_window_seconds: float

    mean_power_w: float
    median_power_w: float
    peak_power_w: float
    energy_j: float

    mean_gpu_util_pct: float
    peak_gpu_util_pct: float

    mean_nvml_memory_bytes: float
    peak_nvml_memory_bytes: int


class NVMLSampler:
    """
    Device-level telemetry sampler.

    Important:
      - Power and memory are for the whole GPU device.
      - Run experiments on an exclusively allocated GPU.
      - Energy is numerical integration of sampled power.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_s: float = 0.02,
    ):
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)

        self._running = False
        self._thread = None
        self._samples = []

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.device_index
        )

    def _sample_once(self):
        t = time.perf_counter()

        try:
            power_w = (
                pynvml.nvmlDeviceGetPowerUsage(
                    self._handle
                )
                / 1000.0
            )
        except pynvml.NVMLError:
            power_w = np.nan

        try:
            util = (
                pynvml.nvmlDeviceGetUtilizationRates(
                    self._handle
                ).gpu
            )
        except pynvml.NVMLError:
            util = np.nan

        try:
            mem = (
                pynvml.nvmlDeviceGetMemoryInfo(
                    self._handle
                ).used
            )
        except pynvml.NVMLError:
            mem = np.nan

        self._samples.append(
            (
                float(t),
                float(power_w),
                float(util),
                float(mem),
            )
        )

    def _loop(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval_s)

    def start(self):
        self._samples = []
        self._running = True

        # One immediate sample prevents short calls from
        # occasionally producing an empty trace.
        self._sample_once()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> TelemetryResult:
        self._running = False

        if self._thread is not None:
            self._thread.join()

        self._sample_once()

        if not self._samples:
            return TelemetryResult(
                0, 0.0,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan,
                np.nan, 0,
            )

        a = np.asarray(
            self._samples,
            dtype=float,
        )

        t = a[:, 0]
        power = a[:, 1]
        util = a[:, 2]
        memory = a[:, 3]

        power_valid = np.isfinite(power)

        if power_valid.sum() >= 2:
            energy_j = float(
                np.trapezoid(
                    power[power_valid],
                    t[power_valid],
                )
            )
        else:
            energy_j = np.nan

        return TelemetryResult(
            n_samples=len(a),

            sample_window_seconds=(
                float(t[-1] - t[0])
                if len(t) >= 2
                else 0.0
            ),

            mean_power_w=float(
                np.nanmean(power)
            ),

            median_power_w=float(
                np.nanmedian(power)
            ),

            peak_power_w=float(
                np.nanmax(power)
            ),

            energy_j=energy_j,

            mean_gpu_util_pct=float(
                np.nanmean(util)
            ),

            peak_gpu_util_pct=float(
                np.nanmax(util)
            ),

            mean_nvml_memory_bytes=float(
                np.nanmean(memory)
            ),

            peak_nvml_memory_bytes=int(
                np.nanmax(memory)
            ),
        )
