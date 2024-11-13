.. raw:: html

  <h1 align="center">
    ShackHartmannAnalysis: A comprehensive sensor library for Shack-Hartmann Wavefront Sensors
  </h1>
  <p align="center">
    <img src="https://img.shields.io/badge/python-v3.12-blue" alt="Python 3.12">
    <a href="https://github.com/yuuutosasa/ShackHartmannAnalysis/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
    </a>
  </p>

ShackHartmannAnalysis is a Python package designed for the analysis of a `Shack-Hartmann Wavefront Sensor <https://en.wikipedia.org/wiki/Shack%E2%80%93Hartmann_wavefront_sensor>`_, 
enabling the measurement and reconstruction of wavefront aberrations directly from raw sensor data. 
The Shack-Hartmann sensor is a widely used, conceptually straightforward wavefront sensor that assesses wavefront distortions, making it ideal for adaptive optics systems in large telescopes. 
These systems adjust images to correct for atmospheric turbulence and maintain image clarity.

**This library provides three distinct methods for calculating shifts within subapertures:**

- Moments: Calculates shifts based on image moments.
- Phase Correlation (PC): Uses phase correlation for shift determination.
- Normalized Cross Correlation (NCC): Determines shifts using normalized cross-correlation.


Example
----------

Here is an example image of Shack-Hartmann wavefront sensor and result, showing the sensor data with the shifts measured in each subaperture, and the respective reconstructed wavefront:

.. raw:: html

  <p align="center">
    <img src="./demo/mainImage.png" alt="Moments Results" width="600">
  </p>
  <p align="center">
    <img src="./demo/output/moments_result.png" alt="Moments Results" width="600">
  </p>
   <p align="center">
    <img src="./demo/output/NCC_result.png" alt="NCC Results" width="600">
  </p>
   <p align="center">
    <img src="./demo/output/PC_result.png" alt="PC Results" width="600">
  </p>

This example was created by the ``example.py`` script in the ``demo`` directory.
This script should provide a good starting point to familiarize yourself with the functionality of hswfs.


License
----------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/yuuutosasa/ShackHartmannAnalysis/blob/master/LICENSE>`_ for details.

Portions of this project are based on the `hswfs <https://github.com/timothygebhard/hswfs>`_ repository by Timothy Gebhard, used under MIT License. See the `LICENSE file <https://github.com/timothygebhard/hswfs/blob/master/LICENSE>`_ for more details.
