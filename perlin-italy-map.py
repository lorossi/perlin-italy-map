import time
import shutil
import logging
import argparse
import subprocess

from pathlib import Path
from math import pi, sin, cos
from opensimplex import OpenSimplex
from scipy.interpolate import interp1d
from PIL import Image, ImageFont, ImageDraw


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
    def __init__(self, destination_size, fps):
        self.saturation = [55, 255]  # inside of this
        self.hue = [165, 240]  # outside of this
        # PARAMETERS
        # min width of a line
        self.min_width = 100
        # y spacing between lines
        self.y_resolution = 40
        # max dh of a line (relative to y uyresolution)
        self.y_scl = 8
        # y quantization in levels
        self.y_levels = 16
        # x spacing between lines
        self.x_resolution = 40
        # number of interpolation
        self.x_levels = 16
        # max x spacing between lines
        self.max_gap = 75
        # pixel width of destination image
        self.destination_size = destination_size
        # noise parameters
        self.noise_scl = 0.005
        self.noise_radius = map(fps, 0, 120, 0, 2.5)
        # line alpha
        self.line_alpha = 5
        # watermark font size
        self.font_size = 40
        # noise initialization
        seed = int(time.time())
        self.noise = OpenSimplex(seed=seed)

    def loadSource(self, path):
        self.source_im = Image.open(path)
        self.hsv_im = self.source_im.convert("HSV")  # HSV - detect italy shape
        self.bw_im = self.source_im.convert("L")  # Black and White - height

        # source image size
        image_width = self.source_im.width
        image_height = self.source_im.height
        # biggest of the two
        biggest = max(image_width, image_height)
        # scale of the destination image
        self.scl = self.destination_size / biggest
        # offset of the destination image
        self.dx = int((self.destination_size - image_width * self.scl) / 2)
        self.dy = int((self.destination_size - image_height * self.scl) / 2)
        # calculate line width
        self.line_width = int(1 / self.scl)

    def loadFont(self, path):
        self.font = ImageFont.truetype(font=path, size=self.font_size)

    def loadLines(self):
        # detect lines from image
        self.lines_coords = []
        for y in range(0, self.hsv_im.height, self.y_resolution):
            started = None

            for x in range(0, self.hsv_im.width, self.x_resolution):
                pixel = pixel_to_hsv(self.hsv_im.getpixel((x, y)))
                # check saturation and hue
                hue = pixel[0]  # hue
                sat = pixel[1]  # saturation

                # check if saturation is in boundaries (to find Italy)
                if (self.saturation[0] < sat <= self.saturation[1]):
                    # check if hue is outside boundaries (to find land)
                    if (hue < self.hue[0] or hue > self.hue[1]):
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
        """
        for line in range(len(self.lines_coords) - 1):
            y = self.lines_coords[line]["start"]["y"]
            next_y = self.lines_coords[line + 1]["start"]["y"]
            if y == next_y:  # on the same y
                x = self.lines_coords[line]["end"]["x"]
                next_x = self.lines_coords[line+1]["start"]["x"]
                if next_x - x < self.max_gap:
                    self.lines_coords[line]["end"]["x"] = self.lines_coords[line+1]["start"]["x"] - self.x_resolution
        """

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
                    heights.append(self.bw_im.getpixel((x + dx, y_start)))

                avg_height = int(sum(heights) / len(heights))

                # these coordinates are relative to the new (scaled) image
                self.height_map.append({
                    "start": {
                        "x": x,
                        "y": y_start
                    },
                    "end": {
                        "x": x + self.x_resolution,
                        "y": y_start
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
            height["normalized_height"] = q_height
            # get color (greyscale)
            color = int(map(height["height"], lowest, highest, 0, 255))
            height["color"] = (color, color, color)

    def generateLines(self, percent):
        self.line_coords = []

        # noise angle
        ntheta = percent * 2 * pi
        # noise coordinates
        nx = self.noise_radius * (1 + cos(ntheta))
        ny = self.noise_radius * (1 + sin(ntheta))

        # list of unique y values
        unique_y = list(set(h["start"]["y"] for h in self.height_map))
        for y in unique_y:
            line_x = []
            line_y = []

            lines_to_draw = [line for line in self.height_map if line["start"]["y"] == y]
            for line in range(len(lines_to_draw)):
                # noise coordinates
                nz = lines_to_draw[line]["start"]["x"] * self.noise_scl
                nw = lines_to_draw[line]["start"]["y"] * self.noise_scl
                # noise value
                n = self.noise.noise4d(x=nx, y=ny, z=nz, w=nw)
                # height offset
                ndy = map(n, -1, 1, 0, 2)

                dy = lines_to_draw[line]["normalized_height"]
                line_x.append(lines_to_draw[line]["start"]["x"] + self.dx)
                line_y.append(lines_to_draw[line]["start"]["y"] - dy * ndy + self.dy)

                # this skips gaps (for example, sea)
                if line < len(lines_to_draw) - 1:
                    gap = lines_to_draw[line+1]["start"]["x"] - lines_to_draw[line]["end"]["x"]

                    if gap > self.max_gap:
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
        temp_im = Image.new('RGB', (int(self.source_im.width), int(self.source_im.height)))
        draw = ImageDraw.Draw(temp_im)

        # reset image
        draw.rectangle([0, 0, temp_im.width, temp_im.height], fill=(0, 0, 0))

        for line in self.line_coords:
            if not line:
                continue

            draw.line((line), fill=(255, 255, 255, self.line_alpha), width=self.line_width)

        # hide logo
        #draw.rectangle([7250, 0, 10700, 950], fill=(0, 0, 0))

        new_width = int(self.source_im.width * self.scl)
        new_height = int(self.source_im.height * self.scl)
        temp_im = temp_im.resize((new_width, new_height), Image.ANTIALIAS)

        self.dest_im = Image.new('RGB', (self.destination_size, self.destination_size))
        self.dest_im.paste(temp_im, (self.dx, self.dy))

        # add watermark
        draw = ImageDraw.Draw(self.dest_im)
        text = "Lorenzo Rossi - www.lorenzoros.si"
        x = self.font_size * 0.5
        y = self.destination_size - self.font_size * 1.5
        draw.text((x, y), fill=(255, 255, 255, 32), text=text, font=self.font)

    def saveDestImage(self, folder, filename, frame_num):
        ext = "png"
        number = str(frame_num).zfill(7)
        path = f"{folder}/{filename}_{number}.{ext}"
        self.dest_im.save(path)
        return path

    def checkPause(self):
        printed = False
        while Path("PAUSE").is_file():
            if not printed:
                logging.info("File PAUSE detected. Pausing until it's deleted.")
                printed = True
            time.sleep(1)

        if printed:
            logging.info("Resuming....")


def main():
    parser = argparse.ArgumentParser(description="Generate a looping animation"
                                                 " of Italian mountains")
    parser.add_argument("-d", "--duration", type=int,
                        help="destination video duration (defaults to 15)",
                        default=15)
    parser.add_argument("-f", "--fps", type=int,
                        help="destination video fps (defaults to 60)",
                        default=60)
    parser.add_argument("-s", "--size", type=int,
                        help="destination video width and height in pixel "
                        "(defaults to 1200)",
                        default=1200)
    parser.add_argument("-l", "--log", action="store",
                        choices=["file", "console"], default="file",
                        help="log destination (defaults to file)")
    args = parser.parse_args()

    if args.log == "file":
        logfile = __file__.replace(".py", ".log")
        print(f"Logging into file {logfile}\n\n")
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                            level=logging.INFO, filename=logfile,
                            filemode="w+")
        print("Logging in every-color.log")
    else:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                            level=logging.INFO)

    logging.info("Script started")
    logging.info(f"Generating video: {args.fps} fps, {args.duration} duration, {args.size} pixels of size")
    script_start = time.time()

    source_file = "source/Italia_tinitaly.jpg"
    font = "source/Roboto-LightItalic.ttf"
    destination_folder = "frames"
    video_folder = "output"
    destination_file = "italy"

    total_frames = args.fps * args.duration
    m = Map(args.size, args.fps)

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

    logging.info("Loading images...")
    m.loadSource(source_file)
    logging.info("Images loaded")

    logging.info("Loading font ...")
    m.loadFont(font)
    logging.info("Font loaded")

    logging.info("Loading lines...")
    m.loadLines()
    logging.info("Lines loaded")

    logging.info("Calculating heights...")
    m.calculateHeights()
    logging.info("Heights calculated")

    generating_start = time.time()
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
        if percent > 0:
            elapsed = int(time.time() - generating_start)
            elapsed_min = int(elapsed / 60)
            total = elapsed / percent
            remaining = int(total - elapsed)
            remaining_min = int(remaining / 60)

            log_text = (
                    f"Time elapsed: {elapsed}s (~{elapsed_min}min). "
                    f"Remaining: {remaining}s (~{remaining_min} min)."
            )

            logging.info(log_text)

    # generate the output video
    timestamp = int(time.time())
    try:
        options = f"ffmpeg -y -r {args.fps} -i {destination_folder}/{destination_file}_%07d.png -loop 0 {video_folder}/{destination_file}_{timestamp}.mp4"
        subprocess.run(options.split(" "))
        options = f"ffmpeg -y -r {args.fps} -i {destination_folder}/{destination_file}_%07d.png -crf 5 -q:v 10 -c:v libvpx -c:a libvorbis {video_folder}/{destination_file}_{timestamp}.mp4"
        subprocess.run(options.split(" "))
    except Exception as e:
        logging.error(f"Cannot make output video using ffmpeg. Error: {e}")

    elapsed = int(time.time() - script_start)
    elapsed_min = int(elapsed / 60)
    logging.info(f"Script completed. It took {elapsed} seconds (~{elapsed_min} minutes).")


if __name__ == "__main__":
    main()
