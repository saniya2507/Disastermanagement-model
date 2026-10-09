
import os
import numpy as np
import rasterio

from config import PROCESSED_DIR, IN_CHANNELS

FEATURES = os.path.join(
    PROCESSED_DIR, "assam_2022_features_normalized.tif"
)
LABELS = os.path.join(
    PROCESSED_DIR, "assam_2022_flood_mask.tif"
)
VALIDITY = os.path.join(
    PROCESSED_DIR, "assam_2022_valid_pixels.tif"
)


def main():
    for path in (FEATURES, LABELS, VALIDITY):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")

    print("=" * 60)
    print("EARTHSENSE AI — PROCESSED DATA VERIFICATION")
    print("=" * 60)

    with rasterio.open(FEATURES) as x, \
         rasterio.open(LABELS) as y, \
         rasterio.open(VALIDITY) as v:

        print("\n--- Metadata ---")
        print("Feature dimensions:", x.width, "x", x.height)
        print("Feature channels:", x.count)
        print("Label bands:", y.count)
        print("Validity bands:", v.count)

        same_grid = all(
            ds.width == x.width
            and ds.height == x.height
            and ds.crs == x.crs
            and ds.transform == x.transform
            for ds in (y, v)
        )

        print("Same grid:", "PASS" if same_grid else "FAIL")

        if not same_grid:
            raise ValueError("Processed rasters are not aligned.")

        if x.count != IN_CHANNELS:
            raise ValueError(
                f"Expected {IN_CHANNELS} channels, found {x.count}"
            )

        total = x.width * x.height
        valid_input_pixels = 0
        invalid_input_pixels = 0
        flooded_valid = 0
        nonflooded_valid = 0
        invalid_label_count = 0
        valid_values_count = 0
        global_min = float("inf")
        global_max = float("-inf")
        nonfinite_count = 0

        for _, window in x.block_windows(1):
            data = x.read(window=window)
            labels = y.read(1, window=window)
            valid = v.read(1, window=window) == 1

            finite = np.isfinite(data).all(axis=0)
            in_range = (
                (data >= -1e-6) & (data <= 1.000001)
            ).all(axis=0)

            valid_input_pixels += int(valid.sum())
            invalid_input_pixels += int((~valid).sum())
            nonfinite_count += int((~np.isfinite(data)).sum())

            global_min = min(global_min, float(data.min()))
            global_max = max(global_max, float(data.max()))

            # Labels are only assessed where all input channels
            # are valid and labels are binary.
            binary = (labels == 0) | (labels == 1)
            usable = valid & finite & binary

            flooded_valid += int((usable & (labels == 1)).sum())
            nonflooded_valid += int((usable & (labels == 0)).sum())
            invalid_label_count += int((valid & ~binary).sum())
            valid_values_count += int(usable.sum())

            if not in_range.all():
                raise ValueError(
                    "Found normalized feature values outside [0, 1]."
                )

        print("\n--- Normalized features ---")
        print(f"Minimum value: {global_min:.6f}")
        print(f"Maximum value: {global_max:.6f}")
        print(f"Non-finite channel values: {nonfinite_count:,}")
        print("Range check [0, 1]: PASS")

        print("\n--- Validity mask ---")
        print(f"Valid input pixels: {valid_input_pixels:,}")
        print(f"Invalid input pixels: {invalid_input_pixels:,}")
        print(
            f"Valid percentage: "
            f"{100 * valid_input_pixels / total:.2f}%"
        )

        print("\n--- Labels where inputs are valid ---")
        print(f"Usable binary labels: {valid_values_count:,}")
        print(f"Flooded pixels: {flooded_valid:,}")
        print(f"Non-flooded pixels: {nonflooded_valid:,}")
        print(f"Invalid labels at valid input pixels: {invalid_label_count:,}")

        if valid_values_count:
            print(
                f"Flood fraction: "
                f"{100 * flooded_valid / valid_values_count:.2f}%"
            )
            print(
                f"Non-flood fraction: "
                f"{100 * nonflooded_valid / valid_values_count:.2f}%"
            )

        print("\nVerification complete. No files were modified.")


if __name__ == "__main__":
    main()
