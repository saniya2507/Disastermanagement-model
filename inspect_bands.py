
import rasterio

files = {
    "FEATURE STACK": "data/raw/assam_2022_feature_stack.tif",
    "TERRAIN STACK": "data/raw/assam_terrain_stack_30m.tif",
}

for title, path in files.items():
    print(f"\n{'=' * 55}")
    print(title)
    print(f"{'=' * 55}")

    with rasterio.open(path) as src:
        print("Band count:", src.count)

        for i in range(1, src.count + 1):
            description = src.descriptions[i - 1]
            print(f"Band {i:02d}: {description or '(no band description)'}")
