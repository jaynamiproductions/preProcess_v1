from flask import Blueprint, render_template, request
import pandas as pd
import os
from testing.preprocess import PreProcess, scale

views = Blueprint('views',__name__)

@views.route('/',methods=['GET'])
def home():
    return render_template('home.html')