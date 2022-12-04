#!/bin/bash

echo Please enter your sudo password if you are prompted to do so! 

echo Installing dependencies 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python launch.py
