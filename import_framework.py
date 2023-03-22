import numpy as np
import matplotlib.pyplot as plt

import json
import torch
import torchvision
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import Compose

import io
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS