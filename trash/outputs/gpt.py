#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.filters import threshold_sauvola
from skimage.measure import find_contours
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans


def extract_ink(image):
    gray = image.mean(axis=2)
    t = threshold_sauvola(gray, 25)
    ink = gray < t * 0.9
    ink = ndimage.binary_dilation(ink, iterations=2)
    ink = ndimage.binary_closing(ink, iterations=2)
    return ink


def flood_regions(ink):
    labels, _ = ndimage.label(~ink)
    return labels


def assign_colors(image, regions, n_colors):
    lab = rgb2lab(image)
    ids = np.unique(regions)
    ids = ids[ids != 0]

    region_colors = []
    for r in ids:
        region_colors.append(lab[regions == r].mean(axis=0))

    region_colors = np.array(region_colors)
    k = min(n_colors, len(region_colors))
    km = KMeans(n_clusters=k, random_state=42)

    cluster_ids = km.fit_predict(region_colors)

    palette = lab2rgb(km.cluster_centers_.reshape(-1, 1, 3)).reshape(-1, 3)
    palette = np.clip(palette * 255, 0, 255).astype(np.uint8)

    label_map = np.zeros_like(regions)
    for r, c in zip(ids, cluster_ids):
        label_map[regions == r] = c + 1

    return label_map, palette


def svg_from_labels(label_map, palette, w, h):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" fill-rule="evenodd">'
    ]

    for i in range(1, len(palette) + 1):
        mask = label_map == i
        if not mask.any():
            continue

        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            continue

        c = palette[i - 1]
        fill = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

        for cont in contours:
            if len(cont) < 3:
                continue
            pts = " ".join(f"{p[1]:.2f},{p[0]:.2f}" for p in cont)
            svg.append(f'<path d="M {pts} Z" fill="{fill}" stroke="none"/>')

    svg.append("</svg>")
    return "\n".join(svg)


def debug_plot(image, ink, regions, label_map, palette):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    ax[0].imshow(image)
    ax[0].set_title("Original")

    ax[1].imshow(ink, cmap="gray")
    ax[1].set_title("Ink Mask")

    ax[2].imshow(regions, cmap="tab20")
    ax[2].set_title("Flood Regions")

    ax[3].imshow(palette[label_map - 1])
    ax[3].set_title("Colored Regions")

    ax[4].imshow(palette[label_map - 1])
    ax[4].set_title("Final Preview")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--colors", type=int, default=24)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img = np.array(Image.open(args.input).convert("RGB"))
    h, w = img.shape[:2]

    ink = extract_ink(img)
    regions = flood_regions(ink)
    label_map, palette = assign_colors(img, regions, args.colors)
    svg = svg_from_labels(label_map, palette, w, h)

    out = args.output
    if not out:
        out = str(Path(args.input).with_suffix(".svg"))

    Path(out).write_text(svg)

    if args.debug:
        debug_plot(img, ink, regions, label_map, palette)


if __name__ == "__main__":
    main()
