# Image Studio
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Numpy](https://img.shields.io/badge/numpy-2.3.1-blue.svg)](https://www.python.org/)
[![Cython](https://img.shields.io/badge/cython-3.1.3-blue.svg)](https://www.python.org/)


Welcome to **Image Studio** 👋!

**Image Studio** is a pet project of mine where I code a range of different image processing techniques. My main goal here is to challenge myself to use only Numpy and C-like Cython code to implement image processing algorithms that I find interesting.

I have also written a [Wiki](https://github.com/RafaelAmauri/Image-Studio/wiki) for **Image Studio**. I use the Wiki to explain what some of the more complicated implementations do and also give examples of the implemented functionalities.



# Showcase 


## 🎨 Color Palette Conversion

This feature works similarly to digital [Color Grading](https://en.wikipedia.org/wiki/Color_grading). 

Here's how it works:

**1. Quantization with Dithering**: The image is quantized, meaning we reduce the number of colors in it. We also
use dithering to avoid color banding.

**2. Conversion to HSV**: The image is converted to the HSV color model. This is because in HSV the color is associated with
a single channel - the Hue (H) channel, unlike in RGB where color information is spread across 3 different channels. This makes it
way easier to manipulate color information.

**3. Color Mapping**: A [Color LUT](https://en.wikipedia.org/wiki/3D_lookup_table) (Lookup Table) maps each grayscale value to a specific HSV value.


After a **Base Hue** is specified, a **color palette for that Hue will be created on-the-fly**.

It's also possible to specify a **Hue Range**. This controls how much the available hues in the color palette can deviate from the Base Hue. For more information, read the [Wiki Page](https://github.com/RafaelAmauri/Image-Studio/wiki/Color-Palette-Conversion#hue-range)!

### Some examples:



**Original Image, 16 Million Colors**

![City - Original](assets/city_original.png)


**Cyan Color Palette, 32 Colors**

![City - Cyan](assets/city_cyan.png)


**Pink Color Palette, 32 Colors**


![City - Pink](assets/city_pink.png)


**Original Image (Credit to Keith Griego on Artstation), 16 Million Colors**

![Girl - Original](assets/animegirl_original.png)

**Green Color Palette, 32 Colors**

![Girl - Green](assets/animegirl_green.png)

**Blue Color Palette, 32 Colors**

![Girl - Blue](assets/animegirl_blue.png)


## Quantization and Dithering 🟥 🟩 🟦 🟧 🟪

**Quantization** works by reducing the number of colors in an image by grouping similar colors together.  **Quantizing the image without dithering causes major [Color Banding](https://en.wikipedia.org/wiki/Colour_banding)**.

**Dithering** is just a fancy way of doing quantization. It works by quantizing the image and then distributing the pixels in a way that creates the illusion of a wider color palette.

I have implemented two dithering algorithms:

* [Floyd-Steinberg algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)

* [Ordered dithering](https://en.wikipedia.org/wiki/Ordered_dithering)



**Table of Comparisons**

| **Category**         | **Original Image**                                          | **Quantized  Image (4 Colors)**                                   | **Quantized Image (4 Colors) + Dithering (Ordered Dithering)**| **Quantized Image (4 Colors) + Dithering (Floyd-Steinberg Dithering)**|
|----------------------|----------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Dog (Grayscale)**  | ![dog_g](assets/dog_g.png)     | ![dog_g_q4](assets/dog_g_q4.png)             | ![dog_g_q4_d](assets/dog_g_q4_ordered.png)            | ![dog_g_q4_d](assets/dog_g_q4_floyd_steinberg.png) |
| **Dog (RGB)**     | ![dog_rgb](assets/dog_rgb.png)       | ![dog_rgb_q4](assets/dog_rgb_q4.png)           | ![dog_rgb_q4_d_ordered](assets/dog_rgb_q4_ordered.png)         | ![dog_rgb_q4_d_floyd_steinberg](assets/dog_rgb_q4_floyd_steinberg.png) |
| **Duck (Grayscale)**      | ![duck_g](assets/duck_g.png)           | ![duck_g_q4](assets/duck_g_q4.png)                   | ![duck_g_q4_d](assets/duck_g_q4_ordered.png)                  | ![duck_g_q4_d](assets/duck_g_q4_floyd_steinberg.png)                  |
| **Duck (RGB)**      | ![duck_rgb](assets/duck_rgb.png)          | ![duck_rgb_q4](assets/duck_rgb_q4.png)            | ![duck_rgb_q4_d](assets/duck_rgb_q4_ordered.png)          | ![duck_rgb_q4_d](assets/duck_rgb_q4_floyd_steinberg.png)          |


## 🚀 SIMD Acceleration

Working with images often requires performing operations on thousands of pixels. Since images are essentially large arrays of pixel data and the same operations can be applied to many pixels at the same time, this makes them **ideal candidates for [SIMD acceleration](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)**.

Implementing SIMD acceleration often requires **major rewrites and optimizations**, and the process is often not very straightforward, but it is still very much worth it!

The speedup I got after implementing vectorization is massive. For reference, my Ryzen 7 5700X3D can **quantize**, **blur** and **convert the color palette of an 8K image in around 15 seconds**.
Without vectorization, the same operations on the same image took over **20 minutes**. So, in a way, you could say that the code in this repo is **8K-ready**! 😁


## 🏎️ Leveraging C code

In cases where the code has to run sequentially and parallelism or SIMD acceleration are not applicable (for example, in the [Floyd-Steinberg Dithering Algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)), I opted to use **Cython**, which is a superset of Python that lets me use C-level types and performance optimizations while writing in Python-like syntax. Cython transpiles the code into optimized C, automatically handles the conversion between Python and C types, and compiles it into a shared library that can interact seamlessly with the rest of the Python codebase, combining the readability of Python with the raw performance of C.

To illustrate the benefits of using Cython, my Numpy-based implementation of Floyd–Steinberg dithering took ~3 minutes to process a 4K image. With Cython, the same algorithm runs in just ~3 seconds.


## 🏁 Setup
```bash
# 1) Clone
git clone https://github.com/RafaelAmauri/Image-Studio.git

# 2) Install dependencies
pip install -r requirements.txt

# 3) Compile the Cython code
python setup.py build_ext --build-lib=include/effects/dithering

# 4) Run
python3 main.py -i path/to/image

# 5) You can specify what operations you want with the command line.  For example, to quantize an image with 8 colors, you could run
python3 main.py -i path/to/image --quantize 8

# For a full list of all available options, check out [include/parser.py](include/utils/parser.py). This file contains all the valid operations with a help section for each one.

# 6) Check the result :)
The program will automatically save the post-processed image as processed.png
```