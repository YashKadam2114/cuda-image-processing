#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path


def write_pgm(path: Path, width: int, height: int, seed: int) -> None:
    random.seed(seed)
    pixels = bytearray(width * height)

    cx = random.randint(width // 4, 3 * width // 4)
    cy = random.randint(height // 4, 3 * height // 4)
    radius = random.randint(min(width, height) // 8, min(width, height) // 3)

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            base = int(20 + 120 * (x / max(1, width - 1)))
            circle = 100 if abs(dist - radius) < 4 else 0
            checker = 35 if ((x // 16 + y // 16) % 2 == 0) else 0
            noise = random.randint(0, 15)
            pixels[y * width + x] = max(0, min(255, base + circle + checker + noise))

    with path.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of small PGM images for the CUDA assignment.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", default="data/input")
    args = parser.parse_args()

    if args.count < 100:
        raise SystemExit("Use at least 100 images to demonstrate the assignment's 100s-of-small-inputs requirement.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pgm"):
        old.unlink()

    for i in range(args.count):
        write_pgm(out / f"image_{i:03d}.pgm", args.width, args.height, i + 1)

    print(f"Generated {args.count} images of size {args.width}x{args.height} in {out}")


if __name__ == "__main__":
    main()
