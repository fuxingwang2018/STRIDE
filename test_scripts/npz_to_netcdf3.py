#!/usr/bin/env python3
"""
npz_to_netcdf.py — Convert STRIDE NPZ generation output to NetCDF4.

Stacks all cases along a time axis and writes one NetCDF file per product
and array variable.

Usage:
    python npz_to_netcdf.py <samples_dir> [--out <output_dir>] [--product ensemble_members]

Examples:
    # Convert all products from all cases
    python npz_to_netcdf.py /path/to/generation/samples/

    # Convert only ensemble_members product
    python npz_to_netcdf.py /path/to/generation/samples/ --product ensemble_members

    # Specify output directory
    python npz_to_netcdf.py /path/to/generation/samples/ --out ./netcdf_output/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _is_date_dir(d: Path) -> bool:
    """Return True if directory name looks like YYYYMMDD."""
    return d.is_dir() and d.name.isdigit() and len(d.name) == 8


def load_cases(samples_dir: Path) -> list[dict]:
    """
    Scan samples_dir for YYYYMMDD subdirectories (e.g. 20140818/).
    Also falls back to batch_*/case_* subdirectories for backwards compatibility.
    Each directory must contain one or more product NPZ files.
    Returns list of dicts sorted by date.
    """
    # Prefer YYYYMMDD folders; fall back to batch_*_case_* pattern
    date_dirs  = sorted([d for d in samples_dir.iterdir() if _is_date_dir(d)])
    batch_dirs = sorted(samples_dir.glob("batch_*_case_*"))
    candidates = date_dirs if date_dirs else batch_dirs

    if not candidates:
        print(f"No YYYYMMDD or batch_*_case_* directories found in '{samples_dir}'")
        return []

    mode = "YYYYMMDD" if date_dirs else "batch/case"
    print(f"Directory format: {mode}  ({len(candidates)} folders)")

    cases = []
    for case_dir in candidates:
        npz_files = list(case_dir.glob("*.npz"))
        if not npz_files:
            continue

        # Date is the folder name for YYYYMMDD layout; infer otherwise
        date_str = case_dir.name if _is_date_dir(case_dir) else _infer_date_from_json(case_dir)

        products = {}
        for npz_path in sorted(npz_files):
            product_name = npz_path.stem
            arrays = dict(np.load(npz_path, allow_pickle=True))
            products[product_name] = arrays

        cases.append({
            "case_id":  case_dir.name,
            "case_dir": case_dir,
            "date":     date_str,
            "products": products,
        })

    cases.sort(key=lambda c: c["date"])
    print(f"Loaded {len(cases)} case(s)  |  date range: {cases[0]['date']} -> {cases[-1]['date']}")
    return cases


def _infer_date_from_json(case_dir: Path) -> str:
    """Try to read date from JSON sidecar, fall back to directory name."""
    for json_path in case_dir.glob("*.json"):
        try:
            meta = json.loads(json_path.read_text())
            if "date" in meta:
                return str(meta["date"])
        except Exception:
            pass
    return case_dir.name


# ─────────────────────────────────────────────────────────────────────────────
# NetCDF writing
# ─────────────────────────────────────────────────────────────────────────────

def infer_dim_names(shape: tuple, array_name: str) -> list[str]:
    """
    Assign dimension names unique per array to avoid shape conflicts.
    Arrays like cond_dynamic (23x17) and target (92x68) have different
    spatial sizes, so each gets its own y_<name> / x_<name> dimensions.
    """
    # Sanitise array name for use in dimension labels
    safe = array_name.replace("_physical", "").replace("_members", "")
    ndim = len(shape)
    if ndim == 4:
        if "member" in array_name or "generated_members" in array_name:
            return ["time", f"member_{safe}", f"y_{safe}", f"x_{safe}"]
        return ["time", f"channel_{safe}", f"y_{safe}", f"x_{safe}"]
    if ndim == 3:
        return ["time", f"y_{safe}", f"x_{safe}"]
    if ndim == 2:
        return ["time", f"channel_{safe}"]
    if ndim == 1:
        return ["time"]
    return [f"dim{i}_{safe}" for i in range(ndim)]


def write_netcdf(
    stacked: dict[str, np.ndarray],
    dates: list[str],
    out_path: Path,
    product_name: str,
) -> None:
    """Write a dict of stacked arrays to a single NetCDF4 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.title        = f"STRIDE generation output — product: {product_name}"
        ds.created      = datetime.utcnow().isoformat() + "Z"
        ds.n_cases      = len(dates)
        ds.date_range   = f"{min(dates)} to {max(dates)}"
        ds.Conventions  = "CF-1.8"

        # Time dimension — store as strings (YYYYMMDD)
        ds.createDimension("time", len(dates))
        time_var = ds.createVariable("time", str, ("time",))
        time_var[:] = np.array(dates, dtype=object)
        time_var.long_name = "date"
        time_var.units     = "YYYYMMDD"

        # Track which extra dims have been created
        created_dims = {"time"}

        for arr_name, arr in stacked.items():
            dim_names = infer_dim_names(arr.shape, arr_name)

            # Create any new dimensions needed
            for dname, dsize in zip(dim_names, arr.shape):
                if dname not in created_dims:
                    ds.createDimension(dname, dsize)
                    created_dims.add(dname)

            # Create variable
            var = ds.createVariable(
                arr_name, arr.dtype, dim_names,
                zlib=True, complevel=4,   # compress
            )
            var[:] = arr
            var.long_name = arr_name
            var.shape_note = str(arr.shape)

            print(f"  wrote '{arr_name}': shape={arr.shape}  dims={dim_names}")

    print(f"  → saved: {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def convert(samples_dir: Path, out_dir: Path, product_filter: str | None) -> None:
    cases = load_cases(samples_dir)
    if not cases:
        print("No cases found. Check the samples directory.")
        return

    # Collect all product names
    all_products = set()
    for c in cases:
        all_products.update(c["products"].keys())

    if product_filter:
        products_to_write = [product_filter]
        if product_filter not in all_products:
            print(f"⚠️  Product '{product_filter}' not found. Available: {all_products}")
            return
    else:
        products_to_write = sorted(all_products)

    dates = [c["date"] for c in cases]

    for product_name in products_to_write:
        print(f"\n=== Product: {product_name} ===")

        # Collect arrays for this product across all cases
        # Each case contributes one time step per array
        array_names = set()
        for c in cases:
            if product_name in c["products"]:
                array_names.update(c["products"][product_name].keys())

        stacked = {}
        for arr_name in sorted(array_names):
            slices = []
            for c in cases:
                prod = c["products"].get(product_name, {})
                arr = prod.get(arr_name)
                if arr is None:
                    print(f"  ⚠️  '{arr_name}' missing in case {c['case_id']}, skipping product.")
                    slices = []
                    break
                slices.append(arr)

            if slices:
                # Stack: each slice is e.g. (1, 1, 92, 68) → concat → (T, 1, 92, 68)
                stacked[arr_name] = np.concatenate(slices, axis=0)
                print(f"  stacked '{arr_name}': {slices[0].shape} × {len(slices)} → {stacked[arr_name].shape}")

        if not stacked:
            print(f"  No arrays to write for product '{product_name}'.")
            continue

        out_path = out_dir / f"{product_name}.nc"
        write_netcdf(stacked, dates, out_path, product_name)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert STRIDE NPZ outputs to NetCDF4.")
    parser.add_argument("samples_dir", help="Path to the generation/samples/ directory")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: samples_dir/../netcdf/)")
    parser.add_argument("--product", default=None,
                        help="Only convert this product (default: all products)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out) if args.out else samples_dir.parent / "netcdf"

    print(f"samples_dir : {samples_dir}")
    print(f"out_dir     : {out_dir}")
    print(f"product     : {args.product or 'all'}")

    convert(samples_dir, out_dir, args.product)


if __name__ == "__main__":
    main()
