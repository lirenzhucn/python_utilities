"""A module containing a set of utility functions for image/video annotation.
"""


import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_lefttop_pos(position, margin, box_size, canvas_size):
    mw, mh = margin
    cw, ch = canvas_size
    bw, bh = box_size
    if position == 'lefttop':
        return (mw, mh)
    elif position == 'leftbot':
        return (mw, ch-bh-mh)
    elif position == 'righttop':
        return (cw-bw-mw, mh)
    elif position == 'rightbot':
        return (cw-bw-mw, ch-bh-mh)
    else:
        return None


def draw_time_stamp(im, time, unit, template='{time:.0f} {unit:s}',
                    position='lefttop', margin=(5, 5), font_name='Arial',
                    font_size=24, color=(255, 255, 255)):
    time = float(time)
    text = template.format(**locals())
    h, w = im.shape[:2]
    pimg = Image.fromarray(im)
    draw = ImageDraw.Draw(pimg)
    font = ImageFont.truetype(font_name, font_size)
    textSize = font.getsize(text)
    pos = get_lefttop_pos(position, margin, textSize, (w, h))
    draw.text(pos, text, font=font, fill=color)
    return np.array(pimg)


def draw_scale_bar(im, pixel_size, unit, length, position='rightbot',
                   margin=(5, 5), font_name='Arial', font_size=24,
                   line_width=2, line_color=(255, 255, 255)):
    INTERNAL_SPACING = 5
    h, w = im.shape[:2]
    pimg = Image.fromarray(im)
    draw = ImageDraw.Draw(pimg)
    font = ImageFont.truetype(font_name, font_size)
    label = '{:d} {:}'.format(int(length), unit)
    labelSize = font.getsize(label)
    lineWidth = round(length / pixel_size)
    totalSize = (max(lineWidth, labelSize[0]), labelSize[1] + INTERNAL_SPACING)
    pos = get_lefttop_pos(position, margin, totalSize, (w, h))
    draw.line((pos[0]+totalSize[0]//2-lineWidth//2, pos[1],
               pos[0]+totalSize[0]//2+lineWidth//2, pos[1]),
              width=line_width, fill=line_color)
    draw.text((pos[0]+totalSize[0]//2-labelSize[0]//2,
               pos[1]+totalSize[1]-labelSize[1]), label, font=font,
              fill=line_color)
    return np.array(pimg)
