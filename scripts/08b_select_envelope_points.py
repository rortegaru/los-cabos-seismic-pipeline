#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Allow importing config.py from repository root
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_PROJ, DATA_INTERMEDIATE


# ============================================================
# CONFIGURACIÓN
# ============================================================
CSV_PATH = CATALOG_PROJ

OUTPUT_SELECTED_CSV = DATA_INTERMEDIATE / "selected_envelope_points.csv"
OUTPUT_FULLFLAG_CSV = DATA_INTERMEDIATE / "catalog_tule_projection_with_selection.csv"

# filtros opcionales
DEPTH_MAX = 10.0          # km ; usa None para no filtrar
TMIN_DAYS = 0.0           # días desde lluvia
TMAX_DAYS = 200.0         # usa None para no limitar

# tamaño visual
FIGSIZE = (11, 6)
BASE_POINT_SIZE = 28
SELECTED_POINT_SIZE = 90


# ============================================================
# UTILIDADES
# ============================================================
def load_catalog(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    required = ["event_id", "datetime", "depth_km", "r_proj_m", "t_days_since_rain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.dropna(subset=required).copy()

    # Distancia positiva para envolvente
    # Si quieres conservar signo, cambia aquí
    df["distance_km"] = np.abs(df["r_proj_m"]) / 1000.0
    df["t_days"] = df["t_days_since_rain"].astype(float)
    df["t_seconds"] = df["t_days"] * 86400.0

    # Dmin = r^2 / (4 t), con r en m y t en s
    # solo para t > 0
    df["Dmin_m2s"] = np.where(
        df["t_seconds"] > 0,
        (np.abs(df["r_proj_m"]) ** 2) / (4.0 * df["t_seconds"]),
        np.nan
    )

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if DEPTH_MAX is not None:
        out = out[out["depth_km"] <= DEPTH_MAX].copy()

    if TMIN_DAYS is not None:
        out = out[out["t_days"] >= TMIN_DAYS].copy()

    if TMAX_DAYS is not None:
        out = out[out["t_days"] <= TMAX_DAYS].copy()

    out = out.reset_index(drop=False).rename(columns={"index": "original_row"})
    return out


def nearest_point_index(df: pd.DataFrame, x_click: float, y_click: float,
                        xlim: tuple[float, float], ylim: tuple[float, float]) -> int:
    """
    Busca el punto más cercano al clic, pero normalizando por el rango de ejes
    para que el 'más cercano visualmente' sea razonable.
    """
    xr = max(xlim[1] - xlim[0], 1e-12)
    yr = max(ylim[1] - ylim[0], 1e-12)

    dx = (df["t_days"].to_numpy() - x_click) / xr
    dy = (df["distance_km"].to_numpy() - y_click) / yr
    d2 = dx**2 + dy**2
    return int(np.argmin(d2))


def build_selected_table(df_filtered: pd.DataFrame, selected_ids_in_order: list[int]) -> pd.DataFrame:
    rows = []
    for order, idx in enumerate(selected_ids_in_order, start=1):
        row = df_filtered.loc[idx]

        rows.append({
            "selection_order": order,
            "filtered_index": int(idx),
            "original_row": int(row["original_row"]),
            "event_id": row["event_id"],
            "datetime": row["datetime"],
            "t_days_since_rain": row["t_days"],
            "t_seconds": row["t_seconds"],
            "r_proj_m": row["r_proj_m"],
            "distance_km": row["distance_km"],
            "depth_km": row["depth_km"],
            "magnitude": row["magnitude"] if "magnitude" in row.index else np.nan,
            "err_h_m": row["err_h_m"] if "err_h_m" in row.index else np.nan,
            "Dmin_m2s": row["Dmin_m2s"],
        })

    out = pd.DataFrame(rows)
    return out


def save_outputs(df_filtered: pd.DataFrame, df_original: pd.DataFrame, selected_ids_in_order: list[int]) -> None:
    # 1) CSV solo de seleccionados
    df_sel = build_selected_table(df_filtered, selected_ids_in_order)
    df_sel.to_csv(OUTPUT_SELECTED_CSV, index=False)

    # 2) CSV completo con bandera
    selected_event_ids = set(df_filtered.loc[selected_ids_in_order, "event_id"].tolist())

    df_full = df_original.copy()
    df_full["selected_envelope"] = df_full["event_id"].isin(selected_event_ids)

    # opcional: incluir Dmin también en catálogo completo
    if "Dmin_m2s" not in df_full.columns:
        df_full["t_days"] = df_full["t_days_since_rain"].astype(float)
        df_full["t_seconds"] = df_full["t_days"] * 86400.0
        df_full["Dmin_m2s"] = np.where(
            df_full["t_seconds"] > 0,
            (np.abs(df_full["r_proj_m"]) ** 2) / (4.0 * df_full["t_seconds"]),
            np.nan
        )

    df_full.to_csv(OUTPUT_FULLFLAG_CSV, index=False)

    print("\nArchivos guardados:")
    print(f"  - {OUTPUT_SELECTED_CSV}")
    print(f"  - {OUTPUT_FULLFLAG_CSV}")
    print(f"  - puntos seleccionados: {len(selected_ids_in_order)}")


# ============================================================
# INTERACTIVO
# ============================================================
def interactive_selector(df_original: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    selected_ids_in_order: list[int] = []
    selected_ids_set: set[int] = set()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # nube base
    sc_all = ax.scatter(
        df_filtered["t_days"],
        df_filtered["distance_km"],
        s=BASE_POINT_SIZE,
        alpha=0.75,
        label="Eventos"
    )

    # scatter de seleccionados
    sc_sel = ax.scatter([], [], s=SELECTED_POINT_SIZE, marker="o", label="Seleccionados")

    # anotaciones pequeñas
    annotations = []

    def refresh_selected_plot():
        nonlocal annotations

        # borrar anotaciones viejas
        for ann in annotations:
            ann.remove()
        annotations = []

        if selected_ids_in_order:
            xs = df_filtered.loc[selected_ids_in_order, "t_days"].to_numpy()
            ys = df_filtered.loc[selected_ids_in_order, "distance_km"].to_numpy()
            sc_sel.set_offsets(np.column_stack([xs, ys]))

            for n, idx in enumerate(selected_ids_in_order, start=1):
                row = df_filtered.loc[idx]
                ann = ax.annotate(
                    str(n),
                    (row["t_days"], row["distance_km"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )
                annotations.append(ann)
        else:
            sc_sel.set_offsets(np.empty((0, 2)))

        fig.canvas.draw_idle()

    def print_last_selection(idx: int):
        row = df_filtered.loc[idx]
        print(
            f"Seleccionado -> "
            f"order={len(selected_ids_in_order):02d}, "
            f"event_id={row['event_id']}, "
            f"datetime={row['datetime']}, "
            f"t_days={row['t_days']:.4f}, "
            f"distance_km={row['distance_km']:.4f}, "
            f"depth_km={row['depth_km']:.3f}, "
            f"Dmin={row['Dmin_m2s']:.4f} m²/s"
        )

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = nearest_point_index(
            df_filtered,
            x_click=event.xdata,
            y_click=event.ydata,
            xlim=ax.get_xlim(),
            ylim=ax.get_ylim()
        )

        # clic izquierdo = agregar
        if event.button == 1:
            if idx not in selected_ids_set:
                selected_ids_set.add(idx)
                selected_ids_in_order.append(idx)
                refresh_selected_plot()
                print_last_selection(idx)
            else:
                print(f"event_id={df_filtered.loc[idx, 'event_id']} ya estaba seleccionado.")

        # clic derecho = quitar
        elif event.button == 3:
            if idx in selected_ids_set:
                selected_ids_set.remove(idx)
                selected_ids_in_order.remove(idx)
                refresh_selected_plot()
                print(f"Removido event_id={df_filtered.loc[idx, 'event_id']}")
            else:
                print("Ese punto no estaba seleccionado.")

    def on_key(event):
        if event.key == "s":
            save_outputs(df_filtered, df_original, selected_ids_in_order)

        elif event.key == "c":
            selected_ids_in_order.clear()
            selected_ids_set.clear()
            refresh_selected_plot()
            print("Selección limpiada.")

        elif event.key == "q":
            print("Cerrando figura.")
            plt.close(fig)

        elif event.key == "i":
            df_sel = build_selected_table(df_filtered, selected_ids_in_order)
            if len(df_sel) == 0:
                print("No hay puntos seleccionados.")
            else:
                print("\nResumen de seleccionados:")
                print(df_sel[[
                    "selection_order", "event_id", "datetime",
                    "t_days_since_rain", "distance_km", "depth_km", "Dmin_m2s"
                ]].to_string(index=False))

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_xlabel("Time since rain (days)")
    ax.set_ylabel("Projected distance (km)")
    ax.set_title(
        "Interactive envelope selector\n"
        "Left click = add | Right click = remove | s = save | c = clear | i = info | q = quit"
    )
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    df_original = load_catalog(CSV_PATH)
    df_filtered = apply_filters(df_original)

    print("\n=== Catálogo cargado ===")
    print(f"Archivo                : {CSV_PATH}")
    print(f"Eventos originales     : {len(df_original)}")
    print(f"Eventos filtrados      : {len(df_filtered)}")
    print(f"DEPTH_MAX              : {DEPTH_MAX}")
    print(f"TMIN_DAYS              : {TMIN_DAYS}")
    print(f"TMAX_DAYS              : {TMAX_DAYS}")

    if len(df_filtered) == 0:
        print("No quedaron eventos tras los filtros.")
        return

    interactive_selector(df_original, df_filtered)


if __name__ == "__main__":
    main()
