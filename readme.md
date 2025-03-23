# Motion Extraction

So I was bored on Sunday before ftour and made this thing.

## What it does

Shows moving stuff in videos while making everything else gray.

I saw [this post](https://www.instagram.com/p/DD8JRDjywHv) and thought "let me make that".

## How it works

1. Frame buffer with temporal offset
2. Bitwise inversion of pixel values
3. Alpha blending with 0.5 opacity coefficient
4. Background replacement via binary mask thresholding
5. Additive compositing of motion elements

## Using it

Basic:
```
python main.py video.mp4
```

Save the result:
```
python main.py video.mp4 --output result.mp4
```

See slower stuff moving (like clouds):
```
python main.py video.mp4 --offset 30
```

Make it more/less sensitive:
```
python main.py video.mp4 --threshold 20
```

Black background instead of gray:
```
python main.py video.mp4 --bg-color 0,0,0
```

## Requirements

```
opencv-python
numpy
```

## Try it on

- Trees on a windy day
- Cars driving by
- People walking
- Sunsets, rain
## Example

### Original Video

https://github.com/user-attachments/assets/080512ca-82e9-4b40-adb5-511537082e35

### Motion Extracted

https://github.com/user-attachments/assets/ff203542-4b1d-4abf-83da-4f721456783a

