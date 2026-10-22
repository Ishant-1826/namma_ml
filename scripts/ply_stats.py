import sys
import numpy as np

TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
         "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
         "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
         "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}

def read_ply(path):
    props, n, fmt, current = [], 0, None, None
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("latin1").strip()
            if not line:
                raise SystemExit("Bad PLY header (unexpected end of file)")
            parts = line.split()
            if line.startswith("format"):
                fmt = parts[1]
            elif line.startswith("element"):
                current = parts[1]
                if current == "vertex":
                    n = int(parts[2])
            elif line.startswith("property") and current == "vertex":
                if parts[1] == "list":
                    raise SystemExit("List property in vertex element is not supported")
                props.append((parts[2], TYPES[parts[1]]))
            elif line == "end_header":
                break
        offset = f.tell()
    if fmt != "binary_little_endian":
        raise SystemExit(f"Unsupported PLY format: {fmt}")
    return np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset), props, n

path = sys.argv[1]
d, props, n = read_ply(path)
names = [k for k, _ in props]
need = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "f_dc_0"]
missing = [k for k in need if k not in names]
print("file:", path)
print("properties:", len(names), "| missing required:", missing or "none")
n_rest = sum(1 for k in names if k.startswith("f_rest_"))
print("SH rest coefficients:", n_rest, "(45 = degree 3, 24 = degree 2, 9 = degree 1, 0 = degree 0)")
print("gaussians:", n)
if missing:
    raise SystemExit("This does not look like a Gaussian splat PLY.")
xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-d["opacity"].astype(np.float64)))
sc = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1).astype(np.float64)).max(1)
print("bounds min:", xyz.min(0).round(2), " max:", xyz.max(0).round(2))
for i, a in enumerate("xyz"):
    print(f"{a}: 1st-99th percentile", np.percentile(xyz[:, i], [1, 99]).round(2))
print("below y = -0.3 m: %.2f%%" % (100 * (xyz[:, 1] < -0.3).mean()))
print("opacity < 0.05: %.1f%%   < 0.1: %.1f%%" % (100 * (op < 0.05).mean(), 100 * (op < 0.1).mean()))
print("largest axis > 0.5 m: %.2f%%   > 1 m: %.2f%%" % (100 * (sc > 0.5).mean(), 100 * (sc > 1).mean()))
print("NaN/inf in positions:", int((~np.isfinite(xyz)).sum()))