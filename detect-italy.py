from PIL import Image, ImageDraw
import logging
from scipy.interpolate import interp1d


# correct pixel range
def pixel_to_hsv(pixel, decimals=1):
    h = round(pixel[0] / 255 * 360, decimals)
    s = round(pixel[1] / 255 * 100, decimals)
    v = round(pixel[2] / 255 * 100, decimals)
    return (h, s, v)


# translate variables
def map(value, old_min, old_max, new_min, new_max):
    if value < old_min:
        value = old_min
    elif value > old_max:
        value = old_max

    old_width = old_max - old_min
    new_width = new_max - new_min
    value_scaled = float(value - old_min) / float(old_width)
    return new_min + (value_scaled * new_width)


def quantize(value, levels, min_value=0, max_value=100):
    interval = int((max_value - min_value) / levels)
    if interval == 0:
        interval = 1
    for x in range(int(max_value), int(min_value), -interval):
        if x < value:
            return x

    return min_value


# SPLINE INTERPOLATION
def spline(x, y, length):
    new_y = []
    new_x = []

    k = 3
    if k >= len(x):
        k = len(x) - 1
    if k % 2 == 0:
        k -= 1  # only odd numbers
    if k <= 1:
        return None

    for i in range(len(x) * length):
        new_x.append((x[-1] - x[0]) / (len(x) * length) * i + x[0])

    f = interp1d(x, y, kind=k, fill_value="extrapolate")
    new_y = f(new_x)

    spline_coords = []
    for i in range(len(new_x)):
        spline_coords.append((new_x[i], new_y[i]))
    return spline_coords


# PARAMETERS
source = "TINITALY_image/Italia_tinitaly.jpg"
# source = "DEM_italia.png"
dest = "italy.png"
saturation = {"min": 15, "max": 256}  # inside of this
hue = {"min": 165, "max": 240}  # outside of this

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

logging.info("Script started")
lines_coords = []
height_map = []

logging.info("Loading images...")
source_im = Image.open(source)
hsv_im = source_im.convert("HSV")  # HSV - detect italy
bw_im = source_im.convert("L")  # Black and White - get height

# PARAMETERS

min_width = 70
y_resolution = 100
y_scl = 5
y_levels = 32
x_resolution = 50
x_levels = 32
max_gap = 75
scl = 0.25  # the relative size of the destination image
"""min_width = 20
y_resolution = 20
y_scl = 0.75
y_levels = 10
x_resolution = 5
max_gap = 20
scl = 1"""
# PARAMETERS END


logging.info("Images loaded")
dest_im = Image.new('RGB', (int(source_im.width), int(source_im.height)))
logging.info("Destination image created")

# get lines
logging.info("Loading lines...")
for y in range(0, hsv_im.height, y_resolution):
    started = None

    for x in range(0, hsv_im.width, x_resolution):
        pixel = hsv_im.getpixel((x, y))
        pixel = pixel_to_hsv(pixel)
        # check saturation and hue
        if (pixel[1] > saturation["min"] and pixel[1] < saturation["max"] and (pixel[0] < hue["min"] or pixel[0] > hue["max"])):
            if not started:
                started = x

        elif started:
            if x - started > min_width:
                lines_coords.append({
                    "start": {
                        "x": started,
                        "y": y
                    },
                    "end": {
                        "x": x,
                        "y": y
                    }
                })

            started = None
logging.info("Lines loaded")

# fill gaps
logging.info("Filling gaps...")
for line in range(len(lines_coords) - 1):
    if lines_coords[line]["start"]["y"] == lines_coords[line + 1]["start"]["y"]:  # on the same y
        if lines_coords[line+1]["start"]["x"] - lines_coords[line]["end"]["x"] < max_gap:
            lines_coords[line]["end"]["x"] = lines_coords[line+1]["start"]["x"] - x_resolution
logging.info("Gaps filled")

# get heights
logging.info("Calculating heights...")
for line in lines_coords:
    y_start = line["start"]["y"]
    x_start = line["start"]["x"]
    x_end = line["end"]["x"]

    for x in range(x_start, x_end, x_resolution):
        heights = []
        for dx in range(x_resolution):
            for y in range(y_start, y_start + y_resolution):
                if (x + dx < bw_im.width and y < bw_im.height):
                    heights.append(bw_im.getpixel((x + dx, y)))

        avg_height = int(sum(heights) / len(heights))

        # these coordinates are relative to the new (scaled) image
        height_map.append({
            "start": {
                "x": x,
                "y": y
            },
            "end": {
                "x": x + x_resolution,
                "y": y
            },
            "height": avg_height
        })
logging.info("Heights calculated")

# normalize heights
logging.info("Normalizing heights...")
sorted_height_map = sorted(height_map, key=lambda x: x["height"])
lowest = sorted_height_map[0]["height"]  # lowest point on the map
highest = sorted_height_map[-1]["height"]  # highest point on the map

for height in height_map:
    # height delta
    d_height = y_scl * y_resolution
    # normalize height
    n_height = map(height["height"], lowest, highest, 0, d_height)
    # quantize height
    q_height = quantize(n_height, y_levels, 0, d_height)
    color = int(map(height["height"], lowest, highest, 0, 255))
    height["normalized_height"] = q_height

    height["color"] = (color, color, color)
logging.info("Heights normalized...")

# generate lines
logging.info("Generating line coords...")
line_coords = []

unique_y = list(set(height["start"]["y"] for height in height_map))  # list of y values
for y in unique_y:
    line_x = []
    line_y = []

    lines_to_draw = [line for line in height_map if line["start"]["y"] == y]
    for line in range(len(lines_to_draw)):

        dy = lines_to_draw[line]["normalized_height"]
        line_x.append(lines_to_draw[line]["start"]["x"])
        line_y.append(lines_to_draw[line]["start"]["y"] - dy)

        # this skips gaps (for example, sea)
        if line < len(lines_to_draw) - 1 and lines_to_draw[line+1]["start"]["x"] - lines_to_draw[line]["end"]["x"] > max_gap:
            spline_coords = spline(line_x, line_y, x_levels)
            line_coords.append(spline_coords)
            line_x = []
            line_y = []
            continue

    spline_coords = spline(line_x, line_y, x_levels)
    line_coords.append(spline_coords)

    line_x = []
    line_y = []

logging.info("Line coords generated")

# draw lines
logging.info("Drawing lines...")
draw = ImageDraw.Draw(dest_im)
for line in line_coords:
    if not line:
        continue

    width = int(4 / scl)
    draw.line((line), fill=(255, 255, 255), width=width)

new_width = int(source_im.width * scl)
new_height = int(source_im.height * scl)
dest_im = dest_im.resize((new_width, new_height), Image.ANTIALIAS)

logging.info("Lines drawn")

# save image
logging.info("Saving image...")
dest_im.save(dest)
logging.info(f"Image saved. Filename: {dest}")
logging.info("Script ended")
