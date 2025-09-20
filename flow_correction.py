# flow_correction.py

from dataclasses import dataclass
import math
import numbers
import numpy as np
import pandas as pd
from typing import Iterable, Union
import difflib

# ----- allowed choices & validators -----

ALLOWED_METER_TYPES = ["Linear", "Square Root", "Coriolis"]

ALLOWED_CORRECTION_METHODS = [
    "UOP LPG",
    "UOP Naphtha",
    "API Lubricating Oils API -10 to 45",
    "API Fuel Oil API 0 to 37",
    "API Jet Fuel API 37-48",
    "API Gasoline/Jet API 48-52",
    "API Gasoline 52-85",
    "API Crude API 0 to 100",
]

# ----- gas: data-type allowed values -----
ALLOWED_DCS_DATA_TYPES = ["Raw", "Corrected"]

_METER_L2C = {s.casefold(): s for s in ALLOWED_METER_TYPES}
_CORR_L2C  = {s.casefold(): s for s in ALLOWED_CORRECTION_METHODS}
_DCS_L2C = {s.casefold(): s for s in ALLOWED_DCS_DATA_TYPES}

def _validate_choice(name: str, value: str, l2c_map: dict, allowed_display: list) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string. Allowed: {', '.join(allowed_display)}")
    key = value.strip().casefold()
    if key in l2c_map:
        return l2c_map[key]
    suggestion = difflib.get_close_matches(value, allowed_display, n=1, cutoff=0.72)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise ValueError(f"Invalid {name!r}: '{value}'. Allowed: {', '.join(allowed_display)}.{hint}")

# ----- helpers: tag average (matches your notebook) -----

def _norm(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def get_tag_average(df: pd.DataFrame, tag_keyword: Union[str, Iterable[str]]) -> float:
    if isinstance(tag_keyword, (list, tuple, set)):
        tags = [t for t in tag_keyword if str(t).strip()]
    else:
        tags = [tag_keyword]
    tags_norm = [_norm(t) for t in tags if _norm(t)]
    if not tags_norm:
        return float('nan')

    selected_avgs = []
    for key_norm in tags_norm:
        candidates = []
        for c in df.columns:
            try:
                if key_norm in _norm(c):
                    candidates.append(c)
            except Exception:
                continue
        if not candidates:
            continue
        selected = max(candidates, key=lambda c: df[c].notna().sum())
        avg = float(np.nanmean(df[selected].values))
        selected_avgs.append(avg)

    if not selected_avgs:
        return float('nan')
    return float(np.nansum(selected_avgs))

# ----- VCF kernels (Excel-equivalent) -----

def _to_F(t_c: float) -> float:
    return t_c * 1.8 + 32.0

def _vcf_exp(A: float, Tf_minus_60: float, factor: float) -> float:
    return math.exp(-1.0 * A * Tf_minus_60 * (1.0 + factor * A * Tf_minus_60))

def _vcf_api_crude(rho, t_c):
    A = (341.0957 + 0*rho + 0*rho**2) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_api_gasoline_52_85(rho, t_c):
    A = (192.4571 + 0.2438*rho + 0*rho**2) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_api_gasoline_jet_48_52(rho, t_c):
    A = (1489.067 + 0*rho - 0.0018684*(rho**2)) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_api_jet_37_48(rho, t_c):
    A = (330.301 + 0*rho + 0*rho**2) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_api_fuel_0_37(rho, t_c):
    A = (103.872 + 0.2701*rho + 0*rho**2) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_api_lube_m10_45(rho, t_c):
    A = (0 + 0.3488*rho + 0*rho**2) / (rho**2)
    return _vcf_exp(A, _to_F(t_c)-60.0, 0.8)

def _vcf_uop_naphtha(rho, t_c, design_branch: bool):
    Tf = _to_F(t_c)
    if Tf < 200.0:
        coeff = 0.24338 if design_branch else 0.2438
        A = (192.4571 + coeff*rho + 0*rho**2) / (rho**2)
        return _vcf_exp(A, Tf-60.0, 0.8)
    else:
        A = (2027.5112 - 4.0028*rho + 0.00235028*(rho**2)) / (rho**2)
        return _vcf_exp(A, Tf-60.0, 0.6)

def _vcf_uop_lpg(rho, t_c):
    Tf = _to_F(t_c)
    return -1.0 * (10 ** (-1.0 * (2.64641798 * rho / 1000.0 + 1.40583481))) * (Tf - 60.0) + 1.0

_ALLOWED_METHODS = {
    "uop lpg": _vcf_uop_lpg,
    "uop naphtha": _vcf_uop_naphtha,
    "api lubricating oils api -10 to 45": _vcf_api_lube_m10_45,
    "api fuel oil api 0 to 37": _vcf_api_fuel_0_37,
    "api jet fuel api 37-48": _vcf_api_jet_37_48,
    "api gasoline/jet api 48-52": _vcf_api_gasoline_jet_48_52,
    "api gasoline 52-85": _vcf_api_gasoline_52_85,
    "api crude api 0 to 100": _vcf_api_crude,
}

def _compute_vcf(method: str, rho: float, t_c: float, design_branch: bool) -> float:
    m = method.strip().lower()
    if m not in _ALLOWED_METHODS:
        raise ValueError(f"Unknown correction method: {method}")
    if m == "uop naphtha":
        return max(_vcf_uop_naphtha(rho, t_c, design_branch), 0.0)
    if m == "uop lpg":
        return max(_vcf_uop_lpg(rho, t_c), 0.0)
    return max(_ALLOWED_METHODS[m](rho, t_c), 0.0)

# ----- main calc -----

@dataclass
class LiquidMeterConfig:
    meter_type: str
    design_temp_c: float
    dcs_max: float
    design_base_density: float
    fe_max_kgph: float
    design_flowing_density: float
    correction_method: str

    def __post_init__(self):
        # exact strings (case-insensitive); returns canonical names
        self.meter_type = _validate_choice("meter_type", self.meter_type, _METER_L2C, ALLOWED_METER_TYPES)
        self.correction_method = _validate_choice("correction_method", self.correction_method, _CORR_L2C, ALLOWED_CORRECTION_METHODS)

        # numeric sanity
        for fld, val in [
            ("design_temp_c", self.design_temp_c),
            ("dcs_max", self.dcs_max),
            ("design_base_density", self.design_base_density),
            ("fe_max_kgph", self.fe_max_kgph),
            ("design_flowing_density", self.design_flowing_density),
        ]:
            if not isinstance(val, numbers.Real):
                raise ValueError(f"{fld} must be numeric.")
        if self.dcs_max == 0:
            raise ValueError("dcs_max must be > 0.")
        if self.design_flowing_density <= 0:
            raise ValueError("design_flowing_density must be > 0.")
        if self.design_base_density <= 0:
            raise ValueError("design_base_density must be > 0.")

# ----- GAS: main config -----

@dataclass
class GasMeterConfig:
    """
    Inputs (numbers only; units noted here):
      - meter_type: {"Linear","Square Root","Coriolis"} (case-insensitive)
      - design_temp_c: °C (G4)
      - dcs_max: DCS faceplate max (I3)
      - fe_max_kgph: kg/h (L3)
      - design_pressure_barg: barg (E4)
      - z_design: (I4)
      - design_mw: molecular weight (L4)
      - dcs_data_type: {"Raw","Corrected"} (E3)

    Derived (computed here from your Excel):
      - design_std_density_kg_m3 = design_mw / 22.414 / z_design        # (G5)
      - design_flowing_density_kg_m3 = ((E4+1.013)*design_mw) / (0.083144626*(G4+273)) / z_design  # (I5)
    """
    meter_type: str
    design_temp_c: float
    dcs_max: float
    fe_max_kgph: float
    design_pressure_barg: float
    z_design: float
    design_mw: float
    dcs_data_type: str

    # computed fields
    design_std_density_kg_m3: float = 0.0
    design_flowing_density_kg_m3: float = 0.0

    def __post_init__(self):
        # normalize enumerations (same validator style you use for liquid)
        self.meter_type = _validate_choice("meter_type", self.meter_type, _METER_L2C, ALLOWED_METER_TYPES)
        self.dcs_data_type = _validate_choice("dcs_data_type", self.dcs_data_type, _DCS_L2C, ALLOWED_DCS_DATA_TYPES)

        # numeric sanity
        for fld, val in [
            ("design_temp_c", self.design_temp_c),
            ("dcs_max", self.dcs_max),
            ("fe_max_kgph", self.fe_max_kgph),
            ("design_pressure_barg", self.design_pressure_barg),
            ("z_design", self.z_design),
            ("design_mw", self.design_mw),
        ]:
            if not isinstance(val, numbers.Real):
                raise ValueError(f"{fld} must be numeric.")

        if self.dcs_max <= 0:
            raise ValueError("dcs_max must be > 0.")
        if self.fe_max_kgph <= 0:
            raise ValueError("fe_max_kgph must be > 0.")
        if self.z_design <= 0:
            raise ValueError("z_design must be > 0.")
        if self.design_mw <= 0:
            raise ValueError("design_mw must be > 0.")

        # compute design densities (exact Excel forms)
        # G5 = L4/22.414/I4
        self.design_std_density_kg_m3 = (self.design_mw / 22.414) / self.z_design

        # I5 = ( (E4+1.013)*L4 ) / ( 0.083144626*(G4+273) ) / I4
        self.design_flowing_density_kg_m3 = ((self.design_pressure_barg + 1.013) * self.design_mw) / (
            0.083144626 * (self.design_temp_c + 273.0)
        ) / self.z_design

        if self.design_std_density_kg_m3 <= 0 or self.design_flowing_density_kg_m3 <= 0:
            raise ValueError("Computed design densities must be > 0 (check inputs).")
 

def liquid_correct(
    *,
    config: LiquidMeterConfig,
    actual_flow_reading: float,      # K1 (from DCS or get_tag_average)
    actual_temp_c: float,            # L1 (from DCS or manual)
    actual_lab_base_density: float,  # M1 (from lab fn)
    echo_inputs: bool = True
) -> dict:
    """
    Compute liquid flow correction (Excel-equivalent) and return mass/volumetric rates
    plus flowing density. The math exactly mirrors your sheet, including VCF methods
    (UOP/API families) and square-root meter behavior.

    Parameters
    ----------
    config : LiquidMeterConfig
        Validated configuration:
        - meter_type ∈ {"Linear", "Square Root", "Coriolis"}  (case-insensitive)
        - correction_method ∈ {
            "UOP LPG", "UOP Naphtha",
            "API Lubricating Oils API -10 to 45",
            "API Fuel Oil API 0 to 37",
            "API Jet Fuel API 37-48",
            "API Gasoline/Jet API 48-52",
            "API Gasoline 52-85",
            "API Crude API 0 to 100"
          }
        - design_temp_c (°C), dcs_max, design_base_density (kg/m³),
          fe_max_kgph (kg/h), design_flowing_density (kg/m³)
    actual_flow_reading : float
        K1 — Current DCS reading in the same engineering units scaled by `dcs_max`.
    actual_temp_c : float
        L1 — Flowing temperature in °C.
    actual_lab_base_density : float
        M1 — Base (standard) density from lab in kg/m³.
    echo_inputs : bool, default True
        If True, the returned dict **also** echoes key inputs for traceability
        (e.g., 'correction_method', 'meter_type', 'actual_flow_reading', ...).

    Returns
    -------
    dict
        Always includes:
        - actual_flowing_density_kg_m3 : float  # N1 (kg/m³) = VCF_actual * M1
        - density_factor               : float  # O1 = 1.0 if Coriolis else N1 / R4
        - vol_std_m3ph                 : float  # P1 (m³/h at base) = mass_kgph / M1
        - vol_flow_m3ph                : float  # Q1 (m³/h at flowing) = mass_kgph / N1
        - mass_kgph                    : float  # kg/h
        - mass_tonh                    : float  # t/h (kg/h ÷ 1000)
        - design_vcf                   : float  # VCF at (design_base_density, design_temp_c)
        If `echo_inputs=True`, also includes:
        - meter_type, correction_method, design_temp_c, dcs_max,
          design_base_density, fe_max_kgph, design_flowing_density,
          actual_flow_reading, actual_temp_c, actual_lab_base_density

    Formulae (map to your Excel cells)
    ----------------------------------
    1) VCF_actual = f(method, M1, L1)   # UOP/API kernels; UOP Naphtha has branch at 200 °F
    2) N1 = VCF_actual * M1
    3) O1 = 1.0  if meter_type == "Coriolis"  else  (N1 / R4)
    4) mass_kgph = (R3 / P3) * K1 * O1^(0.5 if meter_type == "Square Root" else 1.0)
    5) vol_std_m3ph  = mass_kgph / M1
    6) vol_flow_m3ph = mass_kgph / N1
    7) design_vcf = f(method, P4, N4)   # design base density & design temp

    Raises
    ------
    ValueError
        - If any required density ≤ 0.
        - If configuration is invalid (handled in LiquidMeterConfig.__post_init__).

    Example
    -------
    flow = liquid_correct(
        config=cfg,
        actual_flow_reading=get_tag_average(U_12, "12-FIC-037"),
        actual_temp_c=40.0,
        actual_lab_base_density=543.0,
    )
    flow['vol_std_m3ph']          # numeric result
    flow['correction_method']     # echoed input when echo_inputs=True
    """

    mtype = config.meter_type  # canonical ("Linear" | "Square Root" | "Coriolis")

    # 1) actual flowing density (N1)
    vcf_actual = _compute_vcf(
        config.correction_method, actual_lab_base_density, actual_temp_c, design_branch=False
    )
    actual_flowing_density = vcf_actual * actual_lab_base_density

    # 2) density factor (O1)
    if mtype == "Coriolis":
        density_factor = 1.0
    else:
        density_factor = actual_flowing_density / config.design_flowing_density

    # 3) mass flow (kg/h)
    exponent = 0.5 if mtype == "Square Root" else 1.0
    mass_kgph = (config.fe_max_kgph / config.dcs_max) * actual_flow_reading * (density_factor ** exponent)
    mass_tonh = mass_kgph / 1000.0

    # 4) volumetric flows
    if actual_lab_base_density <= 0 or actual_flowing_density <= 0:
        raise ValueError("Actual densities must be > 0.")
    vol_std_m3ph = mass_kgph / actual_lab_base_density
    vol_flow_m3ph = mass_kgph / actual_flowing_density

    # 5) design VCF (reference)
    design_vcf = _compute_vcf(
        config.correction_method, config.design_base_density, config.design_temp_c, design_branch=True
    )

    result = {
        "actual_flowing_density_kg_m3": actual_flowing_density,
        "density_factor": density_factor,
        "vol_std_m3ph": vol_std_m3ph,
        "vol_flow_m3ph": vol_flow_m3ph,
        "mass_kgph": mass_kgph,
        "mass_tonh": mass_tonh,
        "design_vcf": design_vcf,
    }

    if echo_inputs:
        result.update({
            # from config
            "meter_type": config.meter_type,
            "correction_method": config.correction_method,
            "design_temp_c": config.design_temp_c,
            "dcs_max": config.dcs_max,
            "design_base_density": config.design_base_density,
            "fe_max_kgph": config.fe_max_kgph,
            "design_flowing_density": config.design_flowing_density,
            # actuals
            "actual_flow_reading": actual_flow_reading,
            "actual_temp_c": actual_temp_c,
            "actual_lab_base_density": actual_lab_base_density,
        })

    return result


def gas_correct(
    *,
    config: GasMeterConfig,
    actual_flow_reading: float,   # B1 (DCS reading; same engineering units scaled by dcs_max)
    actual_temp_c: float,         # D1 (°C)
    actual_pressure_barg: float,  # C1 (barg)
    actual_mw: float,             # E1 (molecular weight)
    echo_inputs: bool = True
) -> dict:
    """
    Compute gas flow correction (Excel-equivalent) and return mass/volumetric rates and densities.

    Units (inputs): temp in °C, pressure in barg, mass flow FE in kg/h.
    Meter types: {"Linear","Square Root","Coriolis"} (case-insensitive).
    DCS data type: {"Raw","Corrected"}.

    Excel mapping:
      I1 (actual_flowing_density)  = (C1+1.013)*E1 / (0.083144626*(D1+273)) / I$4
      H1 (actual_standard_density) = E1 / 22.414
      F1 (data_type_factor)        = IF(OR(E$3="Corrected",G$3="Coriolis"),1,(C1+1.01325)*(G$4+273.15)/((E$4+1.01325)*(D1+273.15)))
      G1 (mw_factor)               = IF(G$3="Coriolis",1,E1/L$4)
      J1 (vol_std_nm3ph)           = L1*1000/H1
      K1 (vol_flow_m3ph)           = L1*1000/I1
      L1 (mass_kgph)               = L$3/I$3 * B1 / 1000 * (F1*G1)^( IF(G$3="Square Root",0.5,1) )
    """

    # 1) densities (I1, H1)
    actual_flowing_density = ((actual_pressure_barg + 1.013) * actual_mw) / (
        0.083144626 * (actual_temp_c + 273.0)
    ) / config.z_design
    actual_standard_density = actual_mw / 22.414

    if actual_flowing_density <= 0 or actual_standard_density <= 0:
        raise ValueError("Actual densities must be > 0. Check actual_mw/pressure/temperature.")

    # 2) factors (F1, G1)
    if (config.dcs_data_type == "Corrected") or (config.meter_type == "Coriolis"):
        data_type_factor = 1.0
    else:
        data_type_factor = ((actual_pressure_barg + 1.01325) * (config.design_temp_c + 273.15)) / (
            (config.design_pressure_barg + 1.01325) * (actual_temp_c + 273.15)
        )

    mw_factor = 1.0 if (config.meter_type == "Coriolis") else (actual_mw / config.design_mw)

    # 3) mass flow (L1)
    exponent = 0.5 if (config.meter_type == "Square Root") else 1.0
    mass_kgph = (config.fe_max_kgph / config.dcs_max) * (actual_flow_reading) * ((data_type_factor * mw_factor) ** exponent)
    mass_tonh = mass_kgph / 1000.0

    # 4) volumetric flows (J1, K1)
    vol_std_nm3ph = mass_kgph  / actual_standard_density     # L1*1000/H1
    vol_flow_m3ph = mass_kgph  / actual_flowing_density      # L1*1000/I1

    result = {
        "actual_flowing_density_kg_m3": actual_flowing_density,   # I1
        "actual_standard_density_kg_m3": actual_standard_density,  # H1
        "data_type_factor": data_type_factor,                      # F1
        "mw_factor": mw_factor,                                    # G1
        "vol_std_nm3ph": vol_std_nm3ph,                            # J1
        "vol_flow_m3ph": vol_flow_m3ph,                            # K1
        "mass_kgph": mass_kgph,                                    # L1
        "mass_tonh": mass_tonh,
        # handy references
        "design_std_density_kg_m3": config.design_std_density_kg_m3,
        "design_flowing_density_kg_m3": config.design_flowing_density_kg_m3,
    }

    if echo_inputs:
        result.update({
            "meter_type": config.meter_type,
            "dcs_data_type": config.dcs_data_type,
            "design_temp_c": config.design_temp_c,
            "design_pressure_barg": config.design_pressure_barg,
            "dcs_max": config.dcs_max,
            "fe_max_kgph": config.fe_max_kgph,
            "design_mw": config.design_mw,
            "z_design": config.z_design,
            "actual_flow_reading": actual_flow_reading,
            "actual_temp_c": actual_temp_c,
            "actual_pressure_barg": actual_pressure_barg,
            "actual_mw": actual_mw,
        })

    return result


# ================= Registry helpers (paste at end of flow_correction.py) =================
import json

def load_flowmeter_registry(path: str) -> dict:
    """
    Load the JSON registry that maps meter_id -> {kind, df_key, config, actuals}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Registry root must be an object (meter_id -> spec).")
    return data

def _infer_kind_from_config(cfg: dict) -> str:
    """
    If 'kind' is missing in JSON, infer: 'liquid' if liquid keys exist, else 'gas'.
    """
    if any(k in cfg for k in ("design_base_density", "correction_method")):
        return "liquid"
    if any(k in cfg for k in ("design_mw", "z_design", "design_pressure_barg")):
        return "gas"
    raise ValueError("Cannot infer meter kind from config; please set 'kind' in JSON.")

def _resolve_df(df_map: dict, spec: dict, df_override=None):
    """
    Resolve a DataFrame to read tags from:
    - df_override (if provided) wins;
    - else df_map[ spec['df_key'] ] if present;
    - else None.
    """
    if df_override is not None:
        return df_override
    if not spec:
        return None
    key = (spec.get("df_key") or "").strip()
    if key:
        if not df_map or key not in df_map:
            raise KeyError(f"df_key '{key}' not found in df_map.")
        return df_map[key]
    return None

def _as_float(x, name: str) -> float:
    import numbers, math, numpy as np
    if isinstance(x, numbers.Real):
        # also reject NaN
        if isinstance(x, float) and (math.isnan(x) if not np.isnan else np.isnan(x)):
            raise ValueError(f"{name} is NaN.")
        return float(x)
    if isinstance(x, dict) and "value" in x:
        v = x["value"]
        if v is None:
            raise ValueError(f"{name} value from lab is None.")
        return float(v)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"{name} must be numeric (or dict with 'value'); got {type(x).__name__}")

def _get_by_source(source: dict, *, name: str, df, lab_file, extract_lab_fn):
    """
    Resolve a numeric value from a 'source' dict:
      { "constant": <num> }  OR
      { "tag": "12-FIC-037" } OR { "tags": [...] }  OR
      { "lab": { "sample": "...", "test": "..." } }

    Priority is handled by the caller (we call once with that source).
    Returns float (may be NaN if source is None).
    """
    if source is None:
        return float("nan")

    # 1) constant
    if "constant" in source:
        return _as_float(source["constant"], name)

    # 2) tag(s) from a DataFrame
    if "tag" in source or "tags" in source:
        if df is None:
            raise ValueError(f"'{name}' needs a DataFrame but none was provided (df_key/df_map missing).")
        tag_spec = source.get("tag") if "tag" in source else source.get("tags")
        return float(get_tag_average(df, tag_spec))

    # 3) lab lookup
    if "lab" in source:
        if not extract_lab_fn or not lab_file:
            raise ValueError(f"'{name}' needs lab_file & extract_lab_fn but one/both missing.")
        lab = source["lab"] or {}
        sample = lab.get("sample") or lab.get("sample_connection")
        test   = lab.get("test")   or lab.get("test_name")
        if not sample or not test:
            raise ValueError(f"'{name}' lab source requires 'sample' and 'test' keys.")
        val = extract_lab_fn(lab_file, sample, test)
        try:
            return _as_float(val, name)
        except Exception as e:
            raise ValueError(
                f"Lab value for '{name}' not found or non-numeric. "
                f"sample='{sample}', test='{test}', got={val!r}"
            ) from e

    # nothing matched
    return float("nan")

def compute_from_registry(
    *,
    registry: dict,
    meter_id: str,
    df_map: dict = None,            # {"U_2": U_2, ...}
    lab_file=None,
    extract_lab_fn=None,
    echo_inputs: bool = True,
    df_override=None                # pass a DataFrame directly (bypasses df_key)
) -> dict:
    """
    One-call compute using JSON:
      out = compute_from_registry(reg, "03-FI-002", df_map=df_map, lab_file=lab_file, extract_lab_fn=extract_lab)
    The JSON tells the code where to read each actual (constant / tag(s) / lab).
    """
    spec = registry.get(meter_id)
    if not spec:
        raise KeyError(f"meter_id '{meter_id}' not in registry.")

    cfg = spec.get("config", {}) or {}
    kind = (spec.get("kind") or "").strip().lower() or _infer_kind_from_config(cfg)
    actuals = spec.get("actuals", {}) or {}

    df = _resolve_df(df_map, spec, df_override)

    if kind == "liquid":
        flow_src = actuals.get("flow")
        temp_src = actuals.get("temp")
        bd_src   = actuals.get("base_density")

        flow = _get_by_source(flow_src, name="flow", df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)
        temp = _get_by_source(temp_src, name="temp", df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)
        bd   = _get_by_source(bd_src,   name="base_density", df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)

        import numpy as np
        if not np.isfinite(flow): raise ValueError(f"[{meter_id}] 'flow' unresolved.")
        if not np.isfinite(temp): raise ValueError(f"[{meter_id}] 'temp' unresolved.")
        if not np.isfinite(bd):   raise ValueError(f"[{meter_id}] 'base_density' unresolved.")

        return liquid_correct(
            config=LiquidMeterConfig(**cfg),
            actual_flow_reading=float(flow),
            actual_temp_c=float(temp),
            actual_lab_base_density=float(bd),
            echo_inputs=echo_inputs,
        )

    elif kind == "gas":
        flow_src = actuals.get("flow")
        temp_src = actuals.get("temp")
        pr_src   = actuals.get("press")
        mw_src   = actuals.get("mw")

        flow = _get_by_source(flow_src, name="flow",  df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)
        temp = _get_by_source(temp_src, name="temp",  df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)
        pr   = _get_by_source(pr_src,   name="press", df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)
        mw   = _get_by_source(mw_src,   name="mw",    df=df, lab_file=lab_file, extract_lab_fn=extract_lab_fn)

        import numpy as np
        if not np.isfinite(flow): raise ValueError(f"[{meter_id}] 'flow' unresolved.")
        if not np.isfinite(temp): raise ValueError(f"[{meter_id}] 'temp' unresolved.")
        if not np.isfinite(pr):   raise ValueError(f"[{meter_id}] 'press' unresolved.")
        if not np.isfinite(mw):   raise ValueError(f"[{meter_id}] 'mw' unresolved.")

        return gas_correct(
            config=GasMeterConfig(**cfg),
            actual_flow_reading=float(flow),
            actual_temp_c=float(temp),
            actual_pressure_barg=float(pr),
            actual_mw=float(mw),
            echo_inputs=echo_inputs,
        )

    else:
        raise ValueError(f"[{meter_id}] Unknown kind '{kind}'.")


