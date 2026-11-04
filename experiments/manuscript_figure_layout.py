# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Fit manuscript 3D panels, including the projected axis labels."""

from itertools import product

from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
import numpy as np


def fit_3d_panel(fig, axis, caption, *, box_aspect, legend=None):
    """Center the complete 3D view in the space above its fixed caption."""
    fig.canvas.draw()
    fig.set_layout_engine(None)
    caption.xy = (.5, axis.get_position().y0)
    caption.xycoords = "figure fraction"
    caption.set_in_layout(False)
    if legend is not None:
        legend.set_bbox_to_anchor((.5, .99), transform=fig.transFigure)
        legend.set_in_layout(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    padding = fig.dpi / 18  # Four points around the view and below its legend.
    top = legend.get_window_extent(renderer).y0 if legend is not None else fig.bbox.y1
    target = Bbox.from_extents(padding, caption.get_window_extent(renderer).y1 + padding,
                              fig.bbox.x1 - padding, top - padding)
    # The projected view can extend beyond the square Axes3D viewport.
    for artist in (*axis.lines, *axis.collections):
        artist.set_clip_on(False)

    def content_bounds():
        # Axes3D.get_tightbbox omits projected panes and axis-label extents.
        corners = np.array(list(product(axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d())))
        x, y, _ = proj3d.proj_transform(*corners.T, axis.get_proj())
        points = axis.transData.transform(np.column_stack((x, y)))
        scene = Bbox([points.min(axis=0), points.max(axis=0)])
        labels = [item.label.get_window_extent(renderer)
                  for item in (axis.xaxis, axis.yaxis, axis.zaxis)]
        return Bbox.union([axis.get_tightbbox(renderer), scene, *labels])

    axis.set_anchor("C")
    zoom = 1.1
    for iteration in range(8):
        axis.set_box_aspect(box_aspect, zoom=zoom)
        fig.canvas.draw()
        bounds = content_bounds()
        offset_x = (target.x0 + target.x1 - bounds.x0 - bounds.x1) / 2
        offset_y = (target.y0 + target.y1 - bounds.y0 - bounds.y1) / 2
        position = axis.get_position(original=True)
        axis.set_position(position.translated(offset_x / fig.bbox.width,
                                              offset_y / fig.bbox.height))
        scale = min(target.width / bounds.width, target.height / bounds.height)
        if abs(scale - 1) < .002 or iteration == 7:
            break
        zoom *= scale
    fig.canvas.draw()
    return zoom
