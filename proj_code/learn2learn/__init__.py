#!/usr/bin/env python3

from ._version import __version__

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]}) #neednt be verbose, can replace with an empty list instead


from . import algorithms
from . import data
from . import gym
from . import text
from . import vision
from . import optim
from . import nn
from .utils import *
