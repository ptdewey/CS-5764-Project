#!/bin/zsh

ffmpeg -framerate 2 -i Biclusters/Epoch_%d.png -c:v libx264 -pix_fmt yuv420p Biclusters/output.mp4 -y &
ffmpeg -framerate 2 -i Circles/Epoch_%d.png -c:v libx264 -pix_fmt yuv420p Circles/output.mp4 -y &
ffmpeg -framerate 2 -i Moons/Epoch_%d.png -c:v libx264 -pix_fmt yuv420p Moons/output.mp4 -y &