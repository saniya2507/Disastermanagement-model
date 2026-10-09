
import os
import numpy as np
import rasterio

from config import (
    FEATURE_STACK,
    FLOOD_MASK,
    TERRAIN_STACK,
    PROCESSED_DIR,
    CHANNEL_NAMES,
    CHANNEL_BOUNDS,
    IN_CHANNELS,
)

OUTPUT_FEATURES = os.path.join(
    PROCESSED_DIR, "assam_2022_features_normalized.tif"
)

OUTPUT_LABELS = os.path.join(
    PROCESSED_DIR, "assam_2022_flood_mask.tif"
)

OUTPUT_VALIDITY = os.path.join(
    PROCESSED_DIR, "assam_2022_valid_pixels.tif"
)


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype(np.float32)
        profile = src.profile.copy()
        return data, profile


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading feature and terrain rasters...")

    features, profile = read_raster(FEATURE_STACK)
    terrain, terrain_profile = read_raster(TERRAIN_STACK)

    with rasterio.open(FLOOD_MASK) as src:
        labels = src.read(1, masked=True).astype(np.float32)
        label_profile = src.profile.copy()

    if features.shape[0] != 20 or terrain.shape[0] != 2:
        raise ValueError("Expected 20 feature bands and 2 terrain bands.")

    if features.shape[1:] != terrain.shape[1:]:
        raise ValueError("Feature and terrain dimensions do not match.")

    if len(CHANNEL_NAMES) != IN_CHANNELS:
        raise ValueError("CHANNEL_NAMES does not match IN_CHANNELS.")

    if len(CHANNEL_NAMES) != 22:
        raise ValueError("Expected exactly 22 input channels.")

    # Combine the 20 feature bands and 2 terrain bands.
    all_data = np.ma.concatenate(
        [features, terrain], axis=0
    )

    # A pixel is valid only if every input channel is finite
    # and unmasked.
    valid = (
        ~np.ma.getmaskarray(all_data)
        & np.isfinite(all_data.data)
    ).all(axis=0)

    print("Normalizing channels...")

    normalized = np.zeros(all_data.shape, dtype=np.float32)

    for i, name in enumerate(CHANNEL_NAMES):
        lower, upper = CHANNEL_BOUNDS[name]

        band = all_data.data[i]
        band_valid = (
            ~np.ma.getmaskarray(all_data[i])
            & np.isfinite(band)
        )

        # Clip and scale to [0, 1].
        clipped = np.clip(band, lower, upper)

        normalized[i] = np.where(
            band_valid,
            (clipped - lower) / (upper - lower),
            0.0,
        )

        print(
            f"Band {i + 1:02d}/22: {name:12s} "
            f"valid={band_valid.mean() * 100:.2f}%"
        )

    # Keep the validity mask separate. A normalized zero can
    # be a real value, so zero alone must not mean NoData.
    output_profile = profile.copy()
    output_profile.update(
        count=IN_CHANNELS,
        dtype="float32",
        nodata=None,
        compress="deflate",
        predictor=3,
    )

    print("Saving normalized features...")

    with rasterio.open(OUTPUT_FEATURES, "w", **output_profile) as dst:
        dst.write(normalized)

        for i, name in enumerate(CHANNEL_NAMES, start=1):
            dst.set_band_description(i, name)

    # Preserve the original label values. Do not turn invalid
    # labels into non-flooded labels.
    label_data = labels.data
    label_valid = (
        ~np.ma.getmaskarray(labels)
        & np.isfinite(label_data)
        & ((label_data == 0) | (label_data == 1))
    )

    label_output = np.where(
        label_valid, label_data, 255
    ).astype(np.uint8)

    label_profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(OUTPUT_LABELS, "w", **label_profile) as dst:
        dst.write(label_output, 1)
        dst.set_band_description(1, "flood_label")

    validity_profile = profile.copy()
    validity_profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(OUTPUT_VALIDITY, "w", **validity_profile) as dst:
        dst.write(valid.astype(np.uint8), 1)
        dst.set_band_description(1, "all_22_channels_valid")

    print("\n--- Preprocessing summary ---")
    print(f"Total pixels: {valid.size:,}")
    print(f"Pixels valid in all channels: {valid.sum():,}")
    print(f"Pixels excluded for invalid inputs: {(~valid).sum():,}")
    print(f"Valid input percentage: {valid.mean() * 100:.2f}%")

    print("\nCreated:")
    print(OUTPUT_FEATURES)
    print(OUTPUT_LABELS)
    print(OUTPUT_VALIDITY)

    print("\nPreprocessing completed. Original rasters were not modified.")
    print(
        "IMPORTANT: The label file still contains heuristic-derived "
        "labels; this script does not validate their accuracy."
    )


if __name__ == "__main__":
    main()
