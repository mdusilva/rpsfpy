import os
import re

from setuptools import setup

vre = re.compile("__version__ = \'(.*?)\'")
m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "rpsfpy", "__init__.py")).read()


version = vre.findall(m)[0]

setup(
      name = "rpsfpy",
      version = version,
      packages = ["rpsfpy"],
      description = "A package for GLAO PSF reconstruction",
      author = "Manuel Silva",
      author_email = "madusilva@gmail.com",
      license="GPLv2",
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Astronomy"

      ],
      install_requires = ["numpy", "scipy", "pyfits>=3.1"],
      package_data = {
          '' : ['masknn.fits','aofile.json', 'pcdata/pclib.json', 'docs/rpsf.m.html', 'docs/zernike.m.html', 'docs/chassat.m.html', 'docs/index.html']
      },
      zip_safe=False
)
