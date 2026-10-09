
# ============================================================
# config.py — EarthSense AI: Assam 2022 Flood Susceptibility
# Central configuration for data, preprocessing, and training
# ============================================================

import os

# ── Paths ───────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

FEATURE_STACK = os.path.join(
    RAW_DIR, "assam_2022_feature_stack.tif"
)

FLOOD_MASK = os.path.join(
    RAW_DIR, "assam_2022_flood_mask.tif"
)

TERRAIN_STACK = os.path.join(
    RAW_DIR, "assam_terrain_stack_30m.tif"
)

MODEL_SAVE_PATH = os.path.join(
    EXPORTS_DIR, "best_model.pth"
)


# ── Data Configuration ──────────────────────────────────────
PATCH_SIZE = 256
STRIDE = 192

# Minimum fraction of pixels that must be valid in a patch.
MIN_VALID_RATIO = 0.7

# The feature raster contains 20 bands.
# The terrain raster contributes 2 additional bands.
IN_CHANNELS = 22

N_DISTRICTS = 35


# ── Channel Names ───────────────────────────────────────────
# Order MUST match the actual GeoTIFF band order.
#
# Feature raster:
#   Bands  1–2  : Sentinel-1 VV/VH
#   Bands  3–14 : Sentinel-2 B1–B12 (selected bands)
#   Bands 15–18 : NDVI, NDWI, MNDWI, NBR
#   Bands 19–20 : Elevation, slope
#
# Terrain raster:
#   Band 21 : TWI_PROXY
#   Band 22 : TRI

CHANNEL_NAMES = [
    "S1_VV",
    "S1_VH",

    "S2_B1",
    "S2_B2",
    "S2_B3",
    "S2_B4",
    "S2_B5",
    "S2_B6",
    "S2_B7",
    "S2_B8",
    "S2_B8A",
    "S2_B9",
    "S2_B11",
    "S2_B12",

    "NDVI",
    "NDWI",
    "MNDWI",
    "NBR",

    "elevation",
    "slope",

    "TWI_PROXY",
    "TRI",
]


# ── Per-channel Normalization Bounds ────────────────────────
# These are initial clipping bounds, not measured physical
# limits for every pixel. Review outliers before training.
#
# Sentinel-2 values in this export are scaled reflectance-like
# values. The exported data includes some values above 10000;
# these bounds should be validated against the data distribution.
#
# TWI_PROXY is a proxy, NOT a scientifically calculated TWI.
# TRI is a terrain roughness proxy and may exceed 60.

CHANNEL_BOUNDS = {
    # Sentinel-1 backscatter (dB)
    "S1_VV": (-35.0, 5.0),
    "S1_VH": (-40.0, 5.0),

    # Sentinel-2 bands
    "S2_B1":  (0.0, 10000.0),
    "S2_B2":  (0.0, 10000.0),
    "S2_B3":  (0.0, 10000.0),
    "S2_B4":  (0.0, 10000.0),
    "S2_B5":  (0.0, 10000.0),
    "S2_B6":  (0.0, 10000.0),
    "S2_B7":  (0.0, 10000.0),
    "S2_B8":  (0.0, 10000.0),
    "S2_B8A": (0.0, 10000.0),
    "S2_B9":  (0.0, 10000.0),
    "S2_B11": (0.0, 10000.0),
    "S2_B12": (0.0, 10000.0),

    # Spectral indices
    "NDVI":  (-1.0, 1.0),
    "NDWI":  (-1.0, 1.0),
    "MNDWI": (-1.0, 1.0),
    "NBR":   (-1.0, 1.0),

    # Terrain
    "elevation": (-100.0, 2000.0),
    "slope":     (0.0, 90.0),

    # Terrain proxies: provisional bounds
    "TWI_PROXY": (-2.0, 10.0),
    "TRI":       (0.0, 120.0),
}


# ── Training Hyperparameters ───────────────────────────────
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NUM_WORKERS = 0
PIN_MEMORY = False


# ── Loss Configuration ─────────────────────────────────────
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0


# ── Model Configuration ─────────────────────────────────────
CNN_OUT_DIM = 512
GCN_HIDDEN = 256
GCN_OUT_DIM = 128
DROPOUT = 0.3


# ── Spatial Cross-Validation Split ──────────────────────────
# Five north–south row blocks:
# Blocks 0–2: training
# Block 3: validation
# Block 4: testing
#
# Ensure patches do not cross split boundaries or overlap
# between training, validation, and test regions.

N_BLOCKS = 5

TRAIN_BLOCKS = [0, 1, 2]
VAL_BLOCKS = [3]
TEST_BLOCKS = [4]


# ── Device ──────────────────────────────────────────────────
DEVICE = "cpu"


# ── Configuration Validation ────────────────────────────────
assert len(CHANNEL_NAMES) == IN_CHANNELS, (
    f"IN_CHANNELS={IN_CHANNELS}, "
    f"but CHANNEL_NAMES contains {len(CHANNEL_NAMES)} entries."
)

assert len(set(CHANNEL_NAMES)) == len(CHANNEL_NAMES), (
    "CHANNEL_NAMES contains duplicate entries."
)

missing_bounds = set(CHANNEL_NAMES) - set(CHANNEL_BOUNDS)

assert not missing_bounds, (
    f"Missing normalization bounds for: {sorted(missing_bounds)}"
)
