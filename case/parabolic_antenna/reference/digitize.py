"""Extract Figure 16 curves from the publisher PDF's two raster image strips.

Usage: pdfimages -f 19 -l 19 -png PAPER.pdf /tmp/paper-figure
       python -m case.parabolic_antenna.reference.digitize /tmp/paper-figure-002.png /tmp/paper-figure-003.png
Only graph pixels are used; no simulated data enter this extraction.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def extract(top, bottom, destination):
    pixels = np.concatenate((np.asarray(Image.open(top).convert('RGB')),
                             np.asarray(Image.open(bottom).convert('RGB'))), axis=0).astype(float)
    if pixels.shape[:2] != (815, 1467):
        raise ValueError('expected original Figure 16 strips: 1467x408 + 1467x407')
    # Pixel calibration from the original axes: theta [-180,180], gain [5,-40] dB.
    # Both panels share the same vertical mapping.
    axis_top, axis_bottom = 61., 725.
    values = {}
    for plane, left, right in (('E', 138, 683), ('H', 854, 1399)):
        for method, channel in (('vfem', 2), ('meep', 0)):
            crop = pixels[:, left:right+1]
            other = 0 if channel == 2 else 2
            mask = (crop[..., channel] > 140) & (crop[..., channel] > 1.5*crop[..., other]) & (crop[..., 1] < 160)
            mask[:62] = False
            mask[726:] = False
            # Exclude colored line samples in legends, including anti-aliasing.
            legend_left = 415 if plane == 'E' else 1130
            mask[76:139, max(0, legend_left-left):] = False
            x, y = [], []
            for col in range(1, right-left):
                rows = np.flatnonzero(mask[:, col])
                if len(rows):
                    x.append(-180+360*col/(right-left))
                    y.append(5-45*(float(np.median(rows))-axis_top)/(axis_bottom-axis_top))
            values[f'{plane}_{method}_db'] = (np.array(x), np.array(y))
    theta = np.arange(-179., 180., 1.)
    with Path(destination).open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['theta_degrees', *values])
        # Blank where curve is clipped or a dash gap exceeds 4 degrees;
        # do not synthesize deep minima from missing pixels.
        for angle in theta:
            row = [angle]
            for x, y in values.values():
                i = np.searchsorted(x, angle)
                supported = 0 < i < len(x) and x[i]-x[i-1] <= 4
                row.append(float(np.interp(angle, x, y)) if supported else '')
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('top')
    parser.add_argument('bottom')
    parser.add_argument('--output', default=str(Path(__file__).with_name('ban2026_figure16.csv')))
    args = parser.parse_args()
    extract(args.top, args.bottom, args.output)


if __name__ == '__main__':
    main()
