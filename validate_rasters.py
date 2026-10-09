
import os
import numpy as np
import rasterio

from config import FEATURE_STACK, FLOOD_MASK, TERRAIN_STACK

FILES = {
    "Feature stack": FEATURE_STACK,
    "Flood mask": FLOOD_MASK,
    "Terrain stack": TERRAIN_STACK,
}


def validate_rasters():
    print("=" * 65)
    print("EARTHSENSE AI — RASTER VALIDATION")
    print("=" * 65)

    # Check that all required files exist.
    for name, path in FILES.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )
        print(f"\n{name}: {path}")

    with rasterio.open(FEATURE_STACK) as features, \
         rasterio.open(FLOOD_MASK) as labels, \
         rasterio.open(TERRAIN_STACK) as terrain:

        # Expected raster dimensions and band counts.
        print("\n--- Raster metadata ---")
        print(
            f"Feature stack: {features.width} x "
            f"{features.height}, {features.count} bands"
        )
        print(
            f"Flood mask:    {labels.width} x "
            f"{labels.height}, {labels.count} band"
        )
        print(
            f"Terrain stack: {terrain.width} x "
            f"{terrain.height}, {terrain.count} bands"
        )

        # Confirm all rasters share the same grid.
        same_grid = all(
            ds.width == features.width
            and ds.height == features.height
            and ds.crs == features.crs
            and ds.transform == features.transform
            for ds in (labels, terrain)
        )

        print("Same grid:", "PASS" if same_grid else "FAIL")
        print("CRS:", features.crs)
        print("Resolution:", features.res)

        if not same_grid:
            raise ValueError(
                "Raster grids do not match. Stop before preprocessing."
            )

        if features.count != 20 or terrain.count != 2:
            raise ValueError(
                "Expected 20 feature bands and 2 terrain bands."
            )

        if labels.count != 1:
            raise ValueError("Expected exactly one flood-mask band.")

        # Pixel-level validity counters.
        total_pixels = features.width * features.height
        valid_all_22 = 0
        valid_features = 0
        valid_terrain = 0
        valid_labels = 0
        flood_pixels = 0
        non_flood_pixels = 0
        invalid_label_values = 0

        # Process in blocks to avoid loading entire rasters into RAM.
        for _, window in features.block_windows(1):
            feature_data = features.read(
                window=window, masked=True
            ).astype(np.float32)

            terrain_data = terrain.read(
                window=window, masked=True
            ).astype(np.float32)

            label_data = labels.read(
                1, window=window, masked=True
            ).astype(np.float32)

            # A pixel is valid only if every channel is finite
            # and not masked by the GeoTIFF.
            feature_valid = (
                ~np.ma.getmaskarray(feature_data)
                & np.isfinite(feature_data.data)
            ).all(axis=0)

            terrain_valid = (
                ~np.ma.getmaskarray(terrain_data)
                & np.isfinite(terrain_data.data)
            ).all(axis=0)

            label_valid = (
                ~np.ma.getmaskarray(label_data)
                & np.isfinite(label_data.data)
            )

            label_values = label_data.data

            # Only 0 and 1 are accepted as binary labels.
            binary_label = (
                label_valid
                & (
                    (label_values == 0)
                    | (label_values == 1)
                )
            )

            valid_features += int(feature_valid.sum())
            valid_terrain += int(terrain_valid.sum())
            valid_labels += int(binary_label.sum())

            valid_all_22 += int(
                (feature_valid & terrain_valid).sum()
            )

            flood_pixels += int(
                (binary_label & (label_values == 1)).sum()
            )

            non_flood_pixels += int(
                (binary_label & (label_values == 0)).sum()
            )

            invalid_label_values += int(
                (label_valid & ~(
                    (label_values == 0)
                    | (label_values == 1)
                )).sum()
            )

        print("\n--- Pixel validity ---")
        print(f"Total pixels: {total_pixels:,}")

        print(
            f"All 20 feature bands valid: "
            f"{valid_features:,} "
            f"({100 * valid_features / total_pixels:.2f}%)"
        )

        print(
            f"Both terrain bands valid: "
            f"{valid_terrain:,} "
            f"({100 * valid_terrain / total_pixels:.2f}%)"
        )

        print(
            f"All 22 input channels valid: "
            f"{valid_all_22:,} "
            f"({100 * valid_all_22 / total_pixels:.2f}%)"
        )

        print("\n--- Flood mask ---")
        print(
            f"Valid binary labels: {valid_labels:,} "
            f"({100 * valid_labels / total_pixels:.2f}%)"
        )
        print(f"Flooded pixels (1): {flood_pixels:,}")
        print(f"Non-flooded pixels (0): {non_flood_pixels:,}")
        print(f"Other finite label values: {invalid_label_values:,}")

        if valid_labels > 0:
            print(
                f"Flooded among valid labels: "
                f"{100 * flood_pixels / valid_labels:.2f}%"
            )
            print(
                f"Non-flooded among valid labels: "
                f"{100 * non_flood_pixels / valid_labels:.2f}%"
            )

        print("\n--- Preliminary assessment ---")

        if valid_all_22 == 0:
            print("FAIL: No pixels have all 22 valid channels.")
        else:
            print("Input validity counts calculated successfully.")

        if invalid_label_values:
            print(
                "WARNING: The mask contains values other than 0 and 1."
            )

        print(
            "NOTE: Valid raster pixels do not prove that flood "
            "labels are accurate ground truth."
        )

        print("\nValidation complete. No files were modified.")


if __name__ == "__main__":
    validate_rasters()
