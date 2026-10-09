
import os
import numpy as np
import rasterio

DATA_DIR = os.path.join("data", "raw")

FILES = {
    "Feature stack": ("assam_2022_feature_stack.tif", 20),
    "Flood mask": ("assam_2022_flood_mask.tif", 1),
    "Terrain stack": ("assam_terrain_stack_30m.tif", 2),
}

print("\n" + "=" * 60)
print("EarthSense AI - GeoTIFF Verification")
print("=" * 60)

results = {}

for label, (filename, expected_bands) in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    print(f"\n--- {label} ---")

    if not os.path.isfile(path):
        print(f"ERROR: File not found: {os.path.abspath(path)}")
        continue

    try:
        with rasterio.open(path) as src:
            results[label] = {
                "shape": (src.count, src.height, src.width),
                "crs": src.crs,
                "transform": src.transform,
            }

            print(f"File:       {filename}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Bands:      {src.count}")
            print(f"CRS:        {src.crs}")
            print(f"Resolution: {src.res}")
            print(f"NoData:     {src.nodata}")

            if src.count == expected_bands:
                print("Band count: PASS")
            else:
                print(
                    f"WARNING: Expected {expected_bands} bands, "
                    f"found {src.count}"
                )

            # Inspect one band at a time to limit memory usage.
            band = src.read(1, masked=True)
            values = band.compressed()
            del band

            if values.size == 0:
                print("ERROR: First band has no valid pixels.")
            else:
                print(f"Valid pixels in band 1: {values.size:,}")
                print(f"Minimum: {values.min():.4f}")
                print(f"Maximum: {values.max():.4f}")

            if label == "Flood mask":
                mask_values = set()

                for _, window in src.block_windows(1):
                    block = src.read(1, window=window, masked=True)
                    valid = block.compressed()

                    if valid.size:
                        mask_values.update(np.unique(valid).tolist())

                print(f"Flood-mask values: {sorted(mask_values)[:30]}")

                if not mask_values:
                    print("WARNING: No valid flood-mask values.")
                elif not set(mask_values).issubset({0, 1}):
                    print(
                        "WARNING: Mask contains values other than "
                        "0 and 1. Check before training."
                    )

    except Exception as exc:
        print(f"ERROR reading file: {exc}")

print("\n" + "=" * 60)
print("Raster alignment checks")
print("=" * 60)

if len(results) == 3:
    reference = results["Feature stack"]

    for label, info in results.items():
        aligned = (
            info["shape"][1:] == reference["shape"][1:]
            and info["crs"] == reference["crs"]
            and info["transform"] == reference["transform"]
        )

        print(f"{label}: {'SAME GRID' if aligned else 'GRID MISMATCH'}")
else:
    print("Alignment check incomplete: one or more files could not be read.")

print("\nVerification finished.")
