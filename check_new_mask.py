
import os
import numpy as np
import rasterio

from config import FEATURE_STACK, FLOOD_MASK

NEW_MASK = os.path.join(
    os.path.dirname(FLOOD_MASK),
    "assam_2022_flood_mask_v2.tif",
)


def main():
    if not os.path.exists(NEW_MASK):
        raise FileNotFoundError(
            f"New flood mask not found:\n{NEW_MASK}\n"
            "Check that the TIFF is in data/raw."
        )

    counts = {0: 0, 1: 0, "other": 0, "masked": 0}
    total = 0

    with rasterio.open(FEATURE_STACK) as features, \
         rasterio.open(NEW_MASK) as mask:

        print("=== NEW FLOOD MASK CHECK ===")
        print("File:", NEW_MASK)
        print("Dimensions:", mask.width, "x", mask.height)
        print("Bands:", mask.count)
        print("CRS:", mask.crs)
        print("Resolution:", mask.res)
        print("NoData value:", mask.nodata)

        same_grid = (
            features.width == mask.width
            and features.height == mask.height
            and features.crs == mask.crs
            and features.transform == mask.transform
        )

        print("Matches feature grid:", "PASS" if same_grid else "FAIL")

        if not same_grid:
            raise ValueError(
                "New mask grid does not match the feature raster."
            )

        if mask.count != 1:
            raise ValueError("Expected one flood-mask band.")

        for _, window in mask.block_windows(1):
            block = mask.read(1, window=window, masked=True)
            values = block.data
            is_masked = np.ma.getmaskarray(block)
            finite = np.isfinite(values)

            counts["masked"] += int(is_masked.sum())
            counts[0] += int(
                ((values == 0) & ~is_masked & finite).sum()
            )
            counts[1] += int(
                ((values == 1) & ~is_masked & finite).sum()
            )
            counts["other"] += int(
                (
                    (values != 0)
                    & (values != 1)
                    & ~is_masked
                    & finite
                ).sum()
            )
            total += values.size

    print("\n--- Pixel counts ---")
    print(f"Total pixels: {total:,}")
    print(f"Valid non-flood labels (0): {counts[0]:,}")
    print(f"Valid flood labels (1): {counts[1]:,}")
    print(f"Masked / NoData pixels: {counts['masked']:,}")
    print(f"Other finite values: {counts['other']:,}")

    valid_binary = counts[0] + counts[1]

    if valid_binary:
        print(
            "Flood percentage among binary labels:",
            f"{100 * counts[1] / valid_binary:.2f}%"
        )

    print("\nCheck complete. No files were modified.")


if __name__ == "__main__":
    main()
