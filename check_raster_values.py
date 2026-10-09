
import os
import numpy as np
import rasterio

DATA_DIR = os.path.join("data", "raw")

FILES = [
    "assam_2022_feature_stack.tif",
    "assam_2022_flood_mask.tif",
    "assam_terrain_stack_30m.tif",
]

for filename in FILES:
    path = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not os.path.isfile(path):
        print("ERROR: File not found:", path)
        continue

    with rasterio.open(path) as src:
        for band_number in range(1, src.count + 1):
            minimum = np.inf
            maximum = -np.inf
            finite_count = 0
            nan_count = 0
            infinite_count = 0
            valid_zero_count = 0

            for _, window in src.block_windows(band_number):
                block = src.read(
                    band_number,
                    window=window,
                    masked=True,
                )

                values = block.compressed()

                if values.size == 0:
                    continue

                finite = np.isfinite(values)
                finite_values = values[finite]

                finite_count += int(finite.sum())
                nan_count += int(np.isnan(values).sum())
                infinite_count += int(np.isinf(values).sum())
                valid_zero_count += int(
                    np.count_nonzero(finite_values == 0)
                )

                if finite_values.size:
                    minimum = min(
                        minimum, float(finite_values.min())
                    )
                    maximum = max(
                        maximum, float(finite_values.max())
                    )

            if finite_count:
                print(
                    f"Band {band_number:02d}: "
                    f"min={minimum:.4f}, "
                    f"max={maximum:.4f}, "
                    f"finite={finite_count:,}, "
                    f"NaN={nan_count:,}, "
                    f"infinite={infinite_count:,}, "
                    f"zeros={valid_zero_count:,}"
                )
            else:
                print(
                    f"Band {band_number:02d}: "
                    "NO FINITE VALUES"
                )

print("\nRaster value inspection complete.")
