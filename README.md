# [wip] senhance

[![PyPI - Version](https://img.shields.io/pypi/v/senhance.svg)](https://pypi.org/project/senhance)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/senhance.svg)](https://pypi.org/project/senhance)

This is a work in progress. I am building this project on my free time mainly to learn about flow matching. Since my expertise lies in audio I have decided to apply it to the task of **speech enhancement**.

At the current time, my appproach is to do **enhancement in the latent** space. I am using audio codecs like Descript Audio Codec or Mimi (from Moshi) to embed audios in that latent space and building a UNet trained with **flow matching**.

## License

`senhance` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
