"""Streamlit analyzer for monochromator-based photodetector EQE scans.

Run with:
    streamlit run photodetector_eqe_analyzer.py
"""

from __future__ import annotations

import base64
import io
import math
import re
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Physical constants (exact SI values)
ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_CONSTANT_JS = 6.62607015e-34
SPEED_OF_LIGHT_MS = 299792458.0

# Fixed measurement settings requested for this setup.
MONO_START_TIME_S = 10.0
DARK_START_TIME_S = 2.0
DARK_END_TIME_S = 9.0
MONO_SWITCH_TIME_S = 2.237
SI_DIODE_AREA_MM2 = 6.0
DEFAULT_DEVICE_AREA_MM2 = 5.0

# WIDECALI_2 Si calibration measured 2026-09-16: 350–850 nm in 1 nm
# increments, stored as wavelength (nm) and EQE (%). The two-column text is
# compressed and embedded so the deployed app remains self-contained.
_DEFAULT_SI_CALIBRATION_ZLIB_B64 = (
    "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"
)
DEFAULT_SI_CALIBRATION = zlib.decompress(
    base64.b64decode(_DEFAULT_SI_CALIBRATION_ZLIB_B64)
).decode("ascii")


@dataclass
class TraceResult:
    name: str
    table: pd.DataFrame
    time_s: np.ndarray
    current_a: np.ndarray
    windows: list[tuple[float, float, float]]
    output_text: str = ""
    output_name: str = ""


def parse_numeric_columns(data: bytes | str, source_name: str) -> pd.DataFrame:
    """Find numerical values in the first two columns, ignoring arbitrary headers."""
    if isinstance(data, bytes):
        text = None
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError(f"Could not decode {source_name}.")
    else:
        text = data

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        fields = [item for item in re.split(r"[,;\t ]+", line.strip()) if item]
        if len(fields) < 2:
            continue
        try:
            x_value = float(fields[0])
            y_value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            rows.append((x_value, y_value))

    if len(rows) < 2:
        raise ValueError(
            f"{source_name} does not contain at least two valid numerical rows "
            "in its first two columns."
        )

    frame = pd.DataFrame(rows, columns=["x", "y"])
    # Average duplicate x values and sort, which also prevents interpolation errors.
    return frame.groupby("x", as_index=False, sort=True)["y"].mean()


def make_wavelengths(start_nm: float, end_nm: float, step_nm: float) -> np.ndarray:
    if step_nm <= 0:
        raise ValueError("Wavelength step must be greater than zero.")
    direction = 1.0 if end_nm >= start_nm else -1.0
    signed_step = direction * step_nm
    span_steps = abs(end_nm - start_nm) / step_nm
    nearest_steps = round(span_steps)
    if not np.isclose(span_steps, nearest_steps, rtol=0, atol=1e-8):
        raise ValueError(
            "The start-to-end wavelength range must be an integer multiple of the step size."
        )
    return start_nm + signed_step * np.arange(nearest_steps + 1, dtype=float)


def extract_trace(
    raw_data: bytes,
    name: str,
    wavelengths_nm: np.ndarray,
    dwell_time_s: float,
    exclusion_percent: float,
) -> TraceResult:
    trace = parse_numeric_columns(raw_data, name)
    time_s = trace["x"].to_numpy(dtype=float)
    current_a = trace["y"].to_numpy(dtype=float)

    dark_mask = (time_s >= DARK_START_TIME_S) & (time_s <= DARK_END_TIME_S)
    if dark_mask.sum() < 2:
        raise ValueError(
            f"{name}: fewer than two data points occur in the fixed dark-current "
            f"window ({DARK_START_TIME_S:g}–{DARK_END_TIME_S:g} s)."
        )
    dark_current_a = float(np.mean(current_a[dark_mask]))

    edge_fraction = exclusion_percent / 100.0
    rows = []
    windows: list[tuple[float, float, float]] = []
    for index, wavelength_nm in enumerate(wavelengths_nm):
        dwell_start = MONO_START_TIME_S + index * (dwell_time_s + MONO_SWITCH_TIME_S)
        average_start = dwell_start + edge_fraction * dwell_time_s
        average_end = dwell_start + (1.0 - edge_fraction) * dwell_time_s
        mask = (time_s >= average_start) & (time_s <= average_end)
        if mask.sum() < 2:
            raise ValueError(
                f"{name}: wavelength {wavelength_nm:g} nm has only {int(mask.sum())} "
                f"point(s) in its averaging window ({average_start:.3f}–"
                f"{average_end:.3f} s). The trace may be truncated or the scan "
                "settings may not match."
            )
        light_current_a = float(np.mean(current_a[mask]))
        photocurrent_a = light_current_a - dark_current_a
        on_off_ratio = (
            abs(light_current_a / dark_current_a)
            if dark_current_a != 0.0
            else np.nan
        )
        rows.append(
            {
                "Wavelength (nm)": wavelength_nm,
                "Dark Current (A)": dark_current_a,
                "Light Current (A)": light_current_a,
                "Photocurrent (A)": photocurrent_a,
                "On/Off Ratio": on_off_ratio,
            }
        )
        windows.append((float(wavelength_nm), average_start, average_end))

    return TraceResult(
        name=name,
        table=pd.DataFrame(rows),
        time_s=time_s,
        current_a=current_a,
        windows=windows,
    )


def interpolate_si_eqe(
    calibration: pd.DataFrame, wavelengths_nm: np.ndarray
) -> np.ndarray:
    cal_wavelength = calibration["x"].to_numpy(dtype=float)
    cal_eqe_percent = calibration["y"].to_numpy(dtype=float)
    low, high = float(cal_wavelength.min()), float(cal_wavelength.max())
    requested_low = float(wavelengths_nm.min())
    requested_high = float(wavelengths_nm.max())
    if requested_low < low or requested_high > high:
        raise ValueError(
            f"The scan range ({requested_low:g}–{requested_high:g} nm) extends "
            f"outside the Si calibration range ({low:g}–{high:g} nm). "
            "Extrapolation is disabled."
        )
    if np.any(cal_eqe_percent <= 0):
        raise ValueError("The Si calibration contains zero or negative EQE values.")
    return np.interp(wavelengths_nm, cal_wavelength, cal_eqe_percent)


def calibrate_reference(
    reference: TraceResult, si_eqe_percent: np.ndarray
) -> pd.DataFrame:
    wavelength_nm = reference.table["Wavelength (nm)"].to_numpy(dtype=float)
    wavelength_m = wavelength_nm * 1e-9
    si_responsivity_aw = (
        (si_eqe_percent / 100.0)
        * ELEMENTARY_CHARGE_C
        * wavelength_m
        / (PLANCK_CONSTANT_JS * SPEED_OF_LIGHT_MS)
    )
    reference_photocurrent_a = np.abs(
        reference.table["Photocurrent (A)"].to_numpy(dtype=float)
    )
    optical_power_w = reference_photocurrent_a / si_responsivity_aw
    irradiance_w_mm2 = optical_power_w / SI_DIODE_AREA_MM2
    if np.any(~np.isfinite(optical_power_w)) or np.any(optical_power_w <= 0):
        raise ValueError(
            "The Si reference produced zero, negative, or non-finite calibrated "
            "optical power at one or more wavelengths."
        )
    return pd.DataFrame(
        {
            "Wavelength (nm)": wavelength_nm,
            "Si EQE (%)": si_eqe_percent,
            "Si Responsivity (A/W)": si_responsivity_aw,
            "Si Photocurrent Magnitude (A)": reference_photocurrent_a,
            "Optical Power on Si Diode (W)": optical_power_w,
            "Irradiance (W/mm^2)": irradiance_w_mm2,
        }
    )


def calculate_device_eqe(
    device: TraceResult, calibration_result: pd.DataFrame, device_area_mm2: float
) -> None:
    wavelength_nm = device.table["Wavelength (nm)"].to_numpy(dtype=float)
    wavelength_m = wavelength_nm * 1e-9
    irradiance_w_mm2 = calibration_result["Irradiance (W/mm^2)"].to_numpy(dtype=float)
    incident_power_w = irradiance_w_mm2 * device_area_mm2
    photocurrent_magnitude_a = np.abs(
        device.table["Photocurrent (A)"].to_numpy(dtype=float)
    )
    responsivity_aw = photocurrent_magnitude_a / incident_power_w
    eqe_percent = (
        responsivity_aw
        * PLANCK_CONSTANT_JS
        * SPEED_OF_LIGHT_MS
        / (ELEMENTARY_CHARGE_C * wavelength_m)
        * 100.0
    )

    # Retain the requested seven-column output order.
    device.table["Responsivity (A/W)"] = responsivity_aw
    device.table["EQE (%)"] = eqe_percent
    device.table = device.table[
        [
            "Wavelength (nm)",
            "Dark Current (A)",
            "Light Current (A)",
            "Photocurrent (A)",
            "Responsivity (A/W)",
            "On/Off Ratio",
            "EQE (%)",
        ]
    ]


def make_output_text(
    result: TraceResult,
    reference_name: str,
    calibration_name: str,
    start_nm: float,
    end_nm: float,
    step_nm: float,
    dwell_time_s: float,
    exclusion_percent: float,
    device_area_mm2: float,
) -> str:
    notes = [
        "# Photodetector EQE processed data",
        f"# Raw trace: {result.name}",
        f"# Si reference trace: {reference_name}",
        f"# Si EQE calibration: {calibration_name}",
        f"# Wavelength scan (nm): start={start_nm:g}, end={end_nm:g}, step={step_nm:g}",
        f"# Dwell time (s): {dwell_time_s:g}",
        f"# Fixed monochromator start time (s): {MONO_START_TIME_S:g}",
        f"# Fixed monochromator switching time (s): {MONO_SWITCH_TIME_S:g}",
        f"# Fixed dark averaging window (s): {DARK_START_TIME_S:g} to {DARK_END_TIME_S:g}",
        f"# Edge exclusion from each side (%): {exclusion_percent:g}",
        f"# Si diode active area (mm^2): {SI_DIODE_AREA_MM2:g}",
        f"# Device active area (mm^2): {device_area_mm2:g}",
        "# Photocurrent is signed: light current - dark current.",
        "# Absolute photocurrent is used for responsivity and EQE.",
    ]
    table_text = result.table.to_csv(sep="\t", index=False, float_format="%.10e")
    return "\n".join(notes) + "\n" + table_text


def processed_filename(raw_name: str) -> str:
    path = Path(raw_name)
    return f"{path.stem}_processed.txt"


def unique_output_names(results: list[TraceResult]) -> None:
    used: set[str] = set()
    for result in results:
        base = processed_filename(result.name)
        candidate = base
        counter = 2
        while candidate.lower() in used:
            candidate = f"{Path(base).stem}_{counter}.txt"
            counter += 1
        used.add(candidate.lower())
        result.output_name = candidate


def build_zip(results: list[TraceResult]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            archive.writestr(result.output_name, result.output_text)
    return buffer.getvalue()


def line_plot(
    results: list[TraceResult], visible_names: list[str], y_column: str, ylabel: str
):
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    visible = [result for result in results if result.name in visible_names]
    for result in visible:
        axis.plot(
            result.table["Wavelength (nm)"],
            result.table[y_column],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=result.name,
        )
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if visible:
        axis.legend(fontsize=8)
        axis.relim()
        axis.autoscale_view()
    else:
        axis.text(0.5, 0.5, "Select at least one dataset", ha="center", va="center")
    figure.tight_layout()
    return figure


def raw_trace_plot(trace: TraceResult):
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(trace.time_s, trace.current_a, color="#303030", linewidth=1.0)
    axis.axvspan(
        DARK_START_TIME_S,
        DARK_END_TIME_S,
        color="#808080",
        alpha=0.18,
        label="Dark average",
    )
    for index, (wavelength, start, end) in enumerate(trace.windows):
        axis.axvspan(
            start,
            end,
            color=plt.cm.viridis(index / max(1, len(trace.windows) - 1)),
            alpha=0.18,
        )
        if len(trace.windows) <= 20 or index % max(1, len(trace.windows) // 10) == 0:
            axis.text(
                (start + end) / 2,
                0.98,
                f"{wavelength:g}",
                rotation=90,
                ha="center",
                va="top",
                fontsize=7,
                transform=axis.get_xaxis_transform(),
            )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Current (A)")
    axis.set_title(trace.name)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def calibration_plot(calibration_result: pd.DataFrame):
    figure, power_axis = plt.subplots(figsize=(8.2, 4.8))
    wavelength = calibration_result["Wavelength (nm)"]
    power_axis.plot(
        wavelength,
        calibration_result["Optical Power on Si Diode (W)"],
        color="#276FBF",
        marker="o",
        markersize=3,
        label="Power on Si diode",
    )
    power_axis.set_xlabel("Wavelength (nm)")
    power_axis.set_ylabel("Optical power (W)", color="#276FBF")
    power_axis.tick_params(axis="y", labelcolor="#276FBF")
    irradiance_axis = power_axis.twinx()
    irradiance_axis.plot(
        wavelength,
        calibration_result["Irradiance (W/mm^2)"],
        color="#D1495B",
        linestyle="--",
        label="Irradiance",
    )
    irradiance_axis.set_ylabel("Irradiance (W/mm²)", color="#D1495B")
    irradiance_axis.tick_params(axis="y", labelcolor="#D1495B")
    power_axis.grid(alpha=0.25)
    lines = power_axis.lines + irradiance_axis.lines
    power_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="Photodetector EQE Analyzer", layout="wide")
    st.title("Photodetector EQE Analyzer")
    st.caption(
        "Convert unsynchronized current–time scans into photocurrent, responsivity, "
        "on/off ratio, and external quantum efficiency."
    )

    with st.sidebar:
        st.header("Monochromator settings")
        start_nm = st.number_input("Start wavelength (nm)", value=400.0, step=1.0)
        end_nm = st.number_input("End wavelength (nm)", value=700.0, step=1.0)
        step_nm = st.number_input(
            "Wavelength step (nm)", min_value=0.001, value=5.0, step=1.0
        )
        dwell_time_s = st.number_input(
            "Dwell time per wavelength (s)", min_value=0.001, value=5.0, step=0.5
        )
        exclusion_percent = st.slider(
            "Exclude from each edge of dwell (%)",
            min_value=0.0,
            max_value=49.0,
            value=10.0,
            step=1.0,
            help="The selected fraction is discarded from both the beginning and end.",
        )
        st.header("Active areas")
        st.number_input(
            "Si diode area (mm²)", value=SI_DIODE_AREA_MM2, disabled=True
        )
        device_area_mm2 = st.number_input(
            "Device active area (mm²)",
            min_value=1e-9,
            value=DEFAULT_DEVICE_AREA_MM2,
            step=0.1,
            format="%.4f",
        )
        with st.expander("Fixed timing settings"):
            st.write(f"Monochromator starts at **{MONO_START_TIME_S:g} s**")
            st.write(
                f"Dark current is averaged from **{DARK_START_TIME_S:g} to "
                f"{DARK_END_TIME_S:g} s**"
            )
            st.write(f"Wavelength switching takes **{MONO_SWITCH_TIME_S:g} s**")

    st.subheader("1. Si EQE calibration")
    use_custom_calibration = st.checkbox("Upload a replacement Si EQE calibration")
    custom_calibration = None
    if use_custom_calibration:
        custom_calibration = st.file_uploader(
            "Si EQE calibration file", type=["txt", "csv", "dat"], key="calibration"
        )
        st.caption(
            "The first two numerical columns must be wavelength (nm) and EQE (%)."
        )
    else:
        st.info(
            "Using the embedded WIDECALI_2 Si calibration measured 2026-09-16 "
            "(350–850 nm, 1 nm spacing, EQE in %)."
        )

    st.subheader("2. Si photodiode reference scan")
    reference_file = st.file_uploader(
        "Upload the Si diode current–time file",
        type=["txt", "csv", "dat"],
        key="reference",
    )

    st.subheader("3. Device scans")
    device_files = st.file_uploader(
        "Upload one or more device current–time files",
        type=["txt", "csv", "dat"],
        accept_multiple_files=True,
        key="devices",
    )

    if st.button("Process data", type="primary", use_container_width=True):
        if reference_file is None:
            st.error("Upload a Si photodiode reference scan.")
        elif not device_files:
            st.error("Upload at least one device scan.")
        elif use_custom_calibration and custom_calibration is None:
            st.error("Upload the replacement Si EQE calibration or disable that option.")
        else:
            try:
                wavelengths_nm = make_wavelengths(start_nm, end_nm, step_nm)
                if custom_calibration is not None:
                    calibration_bytes = custom_calibration.getvalue()
                    calibration_name = custom_calibration.name
                else:
                    calibration_bytes = DEFAULT_SI_CALIBRATION
                    calibration_name = "Embedded WIDECALI_2 Si EQE calibration (2026-09-16)"
                calibration = parse_numeric_columns(calibration_bytes, calibration_name)
                si_eqe_percent = interpolate_si_eqe(calibration, wavelengths_nm)

                reference = extract_trace(
                    reference_file.getvalue(),
                    reference_file.name,
                    wavelengths_nm,
                    dwell_time_s,
                    exclusion_percent,
                )
                reference_calibration = calibrate_reference(reference, si_eqe_percent)

                results: list[TraceResult] = []
                errors: list[str] = []
                for uploaded_file in device_files:
                    try:
                        result = extract_trace(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                            wavelengths_nm,
                            dwell_time_s,
                            exclusion_percent,
                        )
                        calculate_device_eqe(result, reference_calibration, device_area_mm2)
                        results.append(result)
                    except Exception as exc:  # Continue processing other device files.
                        errors.append(str(exc))

                if not results:
                    raise ValueError("No device files could be processed. " + " ".join(errors))

                unique_output_names(results)
                for result in results:
                    result.output_text = make_output_text(
                        result,
                        reference_file.name,
                        calibration_name,
                        start_nm,
                        end_nm,
                        step_nm,
                        dwell_time_s,
                        exclusion_percent,
                        device_area_mm2,
                    )
                st.session_state["eqe_results"] = results
                st.session_state["eqe_reference"] = reference
                st.session_state["eqe_calibration"] = reference_calibration
                st.session_state["eqe_errors"] = errors
                st.success(f"Processed {len(results)} device dataset(s).")
            except Exception as exc:
                st.error(str(exc))

    results = st.session_state.get("eqe_results")
    if not results:
        return

    errors = st.session_state.get("eqe_errors", [])
    for error in errors:
        st.warning(error)

    st.divider()
    st.header("Preview")
    names = [result.name for result in results]
    visible_names = st.multiselect(
        "Datasets shown in preview plots",
        options=names,
        default=names,
        help="This selection affects only the plots, not exported data.",
    )

    photocurrent_tab, eqe_tab, raw_tab, calibration_tab, table_tab = st.tabs(
        ["Photocurrent", "EQE", "Raw traces", "Si calibration", "Data tables"]
    )
    with photocurrent_tab:
        figure = line_plot(results, visible_names, "Photocurrent (A)", "Photocurrent (A)")
        st.pyplot(figure, clear_figure=True)
    with eqe_tab:
        figure = line_plot(results, visible_names, "EQE (%)", "EQE (%)")
        st.pyplot(figure, clear_figure=True)
    with raw_tab:
        all_traces = [st.session_state["eqe_reference"]] + results
        trace_labels = [f"Si reference — {all_traces[0].name}"] + [
            f"Device — {result.name}" for result in results
        ]
        selected_label = st.selectbox("Trace to inspect", trace_labels)
        selected_trace = all_traces[trace_labels.index(selected_label)]
        st.caption(
            "Shaded regions are the points used for the dark and wavelength averages. "
            "Unshaded data are retained only for diagnosis."
        )
        figure = raw_trace_plot(selected_trace)
        st.pyplot(figure, clear_figure=True)
    with calibration_tab:
        figure = calibration_plot(st.session_state["eqe_calibration"])
        st.pyplot(figure, clear_figure=True)
        st.dataframe(st.session_state["eqe_calibration"], use_container_width=True)
    with table_tab:
        selected_table_name = st.selectbox("Dataset", names, key="table_dataset")
        selected_result = next(r for r in results if r.name == selected_table_name)
        st.dataframe(selected_result.table, use_container_width=True)

    st.header("Download processed data")
    st.download_button(
        "Download all processed files (.zip)",
        data=build_zip(results),
        file_name="photodetector_eqe_processed.zip",
        mime="application/zip",
        type="primary",
    )
    columns = st.columns(min(3, len(results)))
    for index, result in enumerate(results):
        with columns[index % len(columns)]:
            st.download_button(
                f"Download {result.output_name}",
                data=result.output_text,
                file_name=result.output_name,
                mime="text/plain",
                key=f"download_{index}_{result.output_name}",
                use_container_width=True,
            )

    with st.expander("Calculation definitions"):
        st.markdown(
            r"""
The signed photocurrent is

$$I_{photo}=I_{light}-I_{dark}.$$

The Si responsivity, calibrated optical power, and irradiance are

$$R_{Si}=\frac{EQE_{Si}}{100}\frac{q\lambda}{hc},\qquad
P_{Si}=\frac{|I_{photo,Si}|}{R_{Si}},\qquad
E=\frac{P_{Si}}{A_{Si}}.$$

Assuming uniform irradiance, the device receives $P_{device}=EA_{device}$. Its
responsivity and EQE are

$$R_{device}=\frac{|I_{photo,device}|}{P_{device}},\qquad
EQE_{device}=100R_{device}\frac{hc}{q\lambda}.$$

The on/off ratio is $|I_{light}/I_{dark}|$.
"""
        )


if __name__ == "__main__":
    main()
