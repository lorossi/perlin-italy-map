# Perlin Italy Map
`Si magna bè, si vive bè, si sta yeah yeah`

My first experiment in DEM (Digital Elevation Models) using Python.

This script looks for Elevation and animates using a bit of Simplex Noise. It's also perfectly looping over its duration. Spot the cut!
Each time the video will be slightly different as the noise function is seeded with the current epoch.

You will need `ffmpeg` to let the script automatically create a final video. If your OS is not equipped with it, you will need to manually compose a video from all the frames.

## Output
Watch a *low-quality gif* [here on Imgur](https://i.imgur.com/CVV3zUM.mp4) or view an higher quality video [here on Vimeo](https://vimeo.com/490516743) or on my [Instagram profile](https://www.instagram.com/lorossi97/). If you're willing to, clone the repo and look inside the *output* folder.

## Commands
| Command | Description | Defaults | Type |
|---|---|---|---|
| `-h --help` | show help | `none` | - | - |
| `-d --duration` | destination video duration | `15` | `int` |
| `-f --fps` | destination video fps | `60` | `int` |
| `-s --size` | destination video width and height in pixel  | `1200` | `int` |
| `-l --log` | log destination | `file` | `{file, console}` |
| `-o --output` | file output | `italy` | `str` |

All arguments are optionals

## Pause
If, for any reason, you need to pause the script, create a file called `PAUSE` in the working folder. As long as the file is there, the script will be paused.

## Notes
This script doesn't work on Windows because of a strange bug in `scipy`. Furthermore, I'm using `ffmpeg` to create the output video.

Since I mainly use Windows 10, I ran it using WSL.

## Credits
The heightmap is provided by TINITALY DEM and can be found [here](http://tinitaly.pi.ingv.it/). Thanks to *Tarquini S., Isola I., Favalli M., Mazzarini F., Bisson M., Pareschi M. T., Boschi E. (2007). TINITALY/01: a new Triangular Irregular Network of Italy, Annals of Geophysics 50, 407 - 425.*

Roboto Font provided by [Google Fonts (made by Christian Robertson)](https://fonts.google.com/specimen/Roboto)

This project is distributed under Attribution 4.0 International (CC BY 4.0) license.
