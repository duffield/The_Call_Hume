# Vocoder prototype

Experimental audio to multiple 0-255 output "vocoder".


## Install

Note: tested only on Unbuntu 22.04 wih python 3.11

Make a python environment:
```
python -m venv .venv
source .venv/bin/activate
```

Note: for Linux need:
```
$ sudo apt install python3.11-dev libasound2-dev libportaudio2 ffmpeg
```

Note: for Mac need ffmpeg installed:
```
$ brew install ffmpeg
```

Install packages:
```
$ pip install python-dotenv "hume[microphone]" pyaudio numpy pyaudio librosa rich scipy pydub webrtcvad sounddevice pyserial pynput keyboard
```

## Run
To use with your microphone for testing:

```
$ python audio_processor.py --list-devices
```

Note the index of your microphone and use that with `--device` parameter. E.g. if microphone was device 1:

```
$ python audio_processor.py --device 4
```

With Hume.ai:
```
$ python audio_processor.py --device 4 --audio hume --visual list
```
(List visual is the best to see the Hume AI chat output)


```
usage: audio_processor.py [-h] [-n NUM_OUTPUTS] [--visualizer {bar,list,bubble,gpio,none,null}] [--vad] [--filter] [--chain CHAIN] [--audio AUDIO] [-r RATE]
                          [--list-devices] [--device DEVICE]

Audio Processor

options:
  -h, --help            show this help message and exit
  -n, --num_outputs, --num-outputs NUM_OUTPUTS
                        Number of output values
  --visual,--visualizer {bar,list,bubble,gpio,none,null},
                        Visualizer type
  --vad                 Use Voice Activity Detection
  --filter              Apply high-pass filter
  --chain CHAIN         Path to the chain configuration file.
  --audio AUDIO         Path to an audio file (.wav, .mp3) for testing or `hume` to process with hume.ai
  -r, --rate, --sampling_rate RATE
                        Sampling rate for audio stream.
  --list-devices
```

You can load custom chains using a json file:

```json
{
    "frequency": 0.5,
    "volume": 0.5,
    "pulse": {
        "weight": 1.0,
        "pulse_frequency": 0.8,
        "pulse_value": 100
    },
    "volume_random": 0.9
}
```

```
$ python audio_processor.py --config chain_config.json
```

You can also select one of the visualizers (see `visualizers.py`):

```
$ python audio_processor.py --visual list
```



## Install on Pi


```bash
sudo apt-get --yes update
sudo apt-get --yes install libasound2-dev libportaudio2 ffmpeg python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate

pip install python-dotenv "hume[microphone]" pyaudio numpy pyaudio librosa pydub scipy webrtcvad sounddevice rpi-lgpio gpiozero

```

https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio-and-the-40-pin-header

In order to use the GPIO ports, your user must be a member of the gpio group. The default user account is a member by default, but you must add other users manually using the following command:
```
sudo usermod -a -G gpio <username>
```

PWM (pulse-width modulation)
    Software PWM available on all pins
    Hardware PWM available on GPIO12, GPIO13, GPIO18, GPIO19

To see pinouts:
```
$ pinout

Description        : Raspberry Pi 5B rev 1.0
Revision           : d04170
SoC                : BCM2712
RAM                : 8GB
Storage            : MicroSD
USB ports          : 4 (of which 2 USB3)
Ethernet ports     : 1 (1000Mbps max. speed)
Wi-fi              : True
Bluetooth          : True
Camera ports (CSI) : 2
Display ports (DSI): 2

,--------------------------------.
| oooooooooooooooooooo J8   : +====
| 1ooooooooooooooooooo      : |USB2
|  Wi  Pi Model 5B  V1.0  fan +====
|  Fi     +---+      +---+       |
|         |RAM|      |RP1|    +====
||p       +---+      +---+    |USB3
||c      -------              +====
||i        SoC      |c|c J14     |
(        -------  J7|s|s 12 +======
|  J2 bat   uart   1|i|i oo |   Net
| pwr\..|hd|...|hd|o|1|0    +======
`-| |-1o|m0|---|m1|--------------'

J8:
   3V3  (1) (2)  5V    
 GPIO2  (3) (4)  5V    
 GPIO3  (5) (6)  GND   
 GPIO4  (7) (8)  GPIO14
   GND  (9) (10) GPIO15
GPIO17 (11) (12) GPIO18
GPIO27 (13) (14) GND   
GPIO22 (15) (16) GPIO23
   3V3 (17) (18) GPIO24
GPIO10 (19) (20) GND   
 GPIO9 (21) (22) GPIO25
GPIO11 (23) (24) GPIO8 
   GND (25) (26) GPIO7 
 GPIO0 (27) (28) GPIO1 
 GPIO5 (29) (30) GND   
 GPIO6 (31) (32) GPIO12
GPIO13 (33) (34) GND   
GPIO19 (35) (36) GPIO16
GPIO26 (37) (38) GPIO20
   GND (39) (40) GPIO21

J2:
RUN (1)
GND (2)

J7:
COMPOSITE (1)
      GND (2)

J14:
TR01 TAP (1) (2) TR00 TAP
TR03 TAP (3) (4) TR02 TAP
```

For further information, please refer to https://pinout.xyz/pinout/pwm


# Running the cyborg algae

Activate the Python environment if you haven't already:
```bash
$ source .venv/bin/activate
```

Find your mic:

```bash
$ python3.11 audio_processor.py --list-devices
```

Likely it is device 1.

To test mic input:

```bash
$ python3.11 audio_processor.py --device 1    
```

Try with the Hume.ai:

```bash
$ python3.11 audio_processor.py --device 1 --audio hume
```

And connect to the bubblers controlled by the PI's GPIO:

```bash
$ python3.11 audio_processor.py --device 1 --audio hume --visual gpio
```


There is a script that will use that same command and restart the process when it crashes:
(edit the script to change devices, etc)
```bash
$ ./auto_vocoder.sh
```