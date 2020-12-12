import time
import shutil
import logging

from pathlib import Path
from math import pi, sin, cos
from PIL import Image, ImageDraw
from opensimplex import OpenSimplex
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

    k = 5
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


class Map:
    def __init__(self):
        self.saturation = [15, 256]  # inside of this
        self.hue = [165, 240]  # outside of this
        # PARAMETERS
        self.min_width = 70
        self.y_resolution = 100
        self.y_scl = 5
        self.y_levels = 32
        self.x_resolution = 50
        self.x_levels = 32
        self.max_gap = 75
        self.scl = 0.25  # the relative size of the destination image
        self.noise_scl = 1
        self.noise_radius = 1

        seed = int(time.time())
        self.noise = OpenSimplex(seed=seed)

    def loadSource(self, path):
        self.source_im = Image.open(path)
        self.hsv_im = self.source_im.convert("HSV")  # HSV - detect italy shape
        self.bw_im = self.source_im.convert("L")  # Black and White - height

    def loadLines(self):
        # detect lines from image
        self.lines_coords = []
        for y in range(0, self.hsv_im.height, self.y_resolution):
            started = None

            for x in range(0, self.hsv_im.width, self.x_resolution):
                pixel = self.hsv_im.getpixel((x, y))
                pixel = pixel_to_hsv(pixel)
                # check saturation and hue
                sat = pixel[1]  # saturation
                hue = pixel[0]  # hue

                if (self.saturation[0] < sat < self.saturation[1] and (hue < self.hue[0] or hue > self.hue[1])):
                    # this is land, start line
                    if not started:
                        started = x

                elif started:
                    # if started, its time to end since we found sea
                    if x - started > self.min_width:
                        # the line has to be long enough
                        self.lines_coords.append({
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

        # fill gaps between lines
        for line in range(len(self.lines_coords) - 1):
            if self.lines_coords[line]["start"]["y"] == self.lines_coords[line + 1]["start"]["y"]:  # on the same y
                if self.lines_coords[line+1]["start"]["x"] - self.lines_coords[line]["end"]["x"] < self.max_gap:
                    self.lines_coords[line]["end"]["x"] = self.lines_coords[line+1]["start"]["x"] - self.x_resolution

    def calculateHeights(self):
        # calculate heights
        self.height_map = []
        for line in self.lines_coords:
            y_start = line["start"]["y"]
            x_start = line["start"]["x"]
            x_end = line["end"]["x"]

            for x in range(x_start, x_end, self.x_resolution):
                heights = []
                for dx in range(self.x_resolution):
                    for y in range(y_start, y_start + self.y_resolution):
                        if (x + dx < self.bw_im.width and y < self.bw_im.height):
                            heights.append(self.bw_im.getpixel((x + dx, y)))

                avg_height = int(sum(heights) / len(heights))

                # these coordinates are relative to the new (scaled) image
                self.height_map.append({
                    "start": {
                        "x": x,
                        "y": y
                    },
                    "end": {
                        "x": x + self.x_resolution,
                        "y": y
                    },
                    "height": avg_height
                })

        # get highest and lowest points
        sorted_height_map = sorted(self.height_map, key=lambda x: x["height"])
        lowest = sorted_height_map[0]["height"]  # lowest point on the map
        highest = sorted_height_map[-1]["height"]  # highest point on the map

        # normalize the heights
        for height in self.height_map:
            # height delta
            self.d_height = self.y_scl * self.y_resolution
            # normalize height
            n_height = map(height["height"], lowest, highest, 0, self.d_height)
            # quantize height
            q_height = quantize(n_height, self.y_levels, 0, self.d_height)
            color = int(map(height["height"], lowest, highest, 0, 255))
            height["normalized_height"] = q_height

            height["color"] = (color, color, color)

    def generateLines(self, percent):
        self.line_coords = []

        # noise angle
        ntheta = percent * 2 * pi
        # noise coordinates
        nx = self.noise_radius * (1 + cos(ntheta))
        ny = self.noise_radius * (1 + sin(ntheta))

        unique_y = list(set(height["start"]["y"] for height in self.height_map))  # list of y values
        for y in unique_y:
            line_x = []
            line_y = []

            lines_to_draw = [line for line in self.height_map if line["start"]["y"] == y]
            for line in range(len(lines_to_draw)):

                # noise value
                n = self.noise.noise4d(x=nx, y=ny, z=lines_to_draw[line]["start"]["x"] * self.noise_scl, w=lines_to_draw[line]["start"]["y"] * self.noise_scl)
                # height offset
                ndy = map(n, -1, 1, -self.d_height / 4, self.d_height / 4)

                dy = lines_to_draw[line]["normalized_height"]
                line_x.append(lines_to_draw[line]["start"]["x"])
                line_y.append(lines_to_draw[line]["start"]["y"] - dy + ndy)

                # this skips gaps (for example, sea)
                if line < len(lines_to_draw) - 1 and lines_to_draw[line+1]["start"]["x"] - lines_to_draw[line]["end"]["x"] > self.max_gap:
                    spline_coords = spline(line_x, line_y, self.x_levels)
                    self.line_coords.append(spline_coords)
                    line_x = []
                    line_y = []
                    continue

            spline_coords = spline(line_x, line_y, self.x_levels)
            self.line_coords.append(spline_coords)

            line_x = []
            line_y = []

    def drawOutput(self):
        self.dest_im = Image.new('RGB', (int(self.source_im.width), int(self.source_im.height)))

        draw = ImageDraw.Draw(self.dest_im)
        # reset image
        draw.rectangle([0, 0, self.dest_im.width, self.dest_im.height], fill=(0, 0, 0))

        for line in self.line_coords:
            if not line:
                continue

            width = int(4 / self.scl)
            draw.line((line), fill=(255, 255, 255), width=width)

        # hide logo
        draw.rectangle([7250, 0, 10700, 950], fill=(0, 0, 0))

        new_width = int(self.source_im.width * self.scl)
        new_height = int(self.source_im.height * self.scl)
        self.dest_im = self.dest_im.resize((new_width, new_height), Image.ANTIALIAS)

    def saveDestImage(self, folder, filename, frame_num):
        name = filename.split(".")[0]
        ext = filename.split(".")[-1]
        number = str(frame_num).zfill(7)
        path = f"{folder}/{name}_{number}.{ext}"
        self.dest_im.save(path)
        return path

    def checkPause(self):
        printed = False
        while Path("PAUSE").is_file():
            if not printed:
                print("File PAUSE detected. Pausing until it's deleted.")
                printed = True
            time.sleep(1)


def main():
    # FFMPEG command:
    # ffmpeg -y -r 60 -i frames/italy_%07d.png -loop 0 output/video.mp4

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.info("Script started")
    script_start = time.time()

    source_file = "source/Italia_tinitaly.jpg"
    destination_folder = "frames"
    video_folder = "output"
    destination_file = "italy.png"
    fps = 60
    duration = 15

    logging.info("Creating folders...")
    try:
        shutil.rmtree(destination_folder)
    except Exception as e:
        logging.info(f"Folder {destination_folder} does not exist. Error: {e}")
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    try:
        shutil.rmtree(video_folder)
    except Exception as e:
        logging.info(f"Folder {video_folder} does not exist. Error: {e}")
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    logging.info("Folders created")

    total_frames = fps * duration
    m = Map()
    logging.info("Loading images...")
    m.loadSource(source_file)
    logging.info("Images loaded")

    logging.info("Loading lines...")
    m.loadLines()
    logging.info("Lines loaded")

    logging.info("Calculating heights...")
    m.calculateHeights()
    logging.info("Heights calculated")

    for x in range(total_frames):
        percent = x / total_frames

        logging.info("Generating line coords...")
        m.generateLines(percent)
        logging.info("Line coords generated")

        logging.info("Drawing output...")
        m.drawOutput()
        logging.info("Output drawn")

        logging.info("Saving image...")
        path = m.saveDestImage(destination_folder, destination_file, x)
        logging.info(f"Image {x+1}/{total_frames} saved. Location: {path}")

        m.checkPause()

    elasped = int(time.time() - script_start)
    logging.info(f"Script completed. It took {elasped} seconds.")


if __name__ == "__main__":
    main()
