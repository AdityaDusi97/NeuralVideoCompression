#!/bin/bash
# script
cd ../raw
find . -maxdepth 2 -mindepth 2 -type d > dir.txt

while IFS= read -r dir; do 
    # echo "${dir}/image03/data"
    filename=$(echo $dir | cut -d'/' -f 3)
    # echo $filename

    # under raw directory
    mkdir "kitti_${filename}";
    RAWPATH="${dir}/image_03/data";

    cd "kitti_${filename}"
    mkdir uncomp decomp 

    echo "Start processing ${filename} ..........."
    # frames to uncomp.mp4
    ffmpeg -nostdin -r 10  -start_number 0 -i ".${RAWPATH}/%10d.png" -vcodec libx264 -crf 0 -pix_fmt yuv420p -an -vf "crop=1200:360:0:0" uncomp.mp4

    # mp4 to cropped frames
    ffmpeg -nostdin -i uncomp.mp4 -framerate 10 -start_number 0 uncomp/%05d.png

    # encode/compress uncomp.mp4 to comp.h264 with specified bitrate/crf
    ffmpeg -nostdin -i uncomp.mp4 -an -vcodec libx264 -maxrate 0.5M -bufsize 0.25M comp.h264

    # decode comp.h264 to decomp.mp4
    ffmpeg -nostdin -framerate 10 -i comp.h264 -vf "fieldorder=bff" decomp.mp4

    # decomp.mp4 to frames
    ffmpeg -nostdin -i decomp.mp4 -framerate 10 -start_number 0 decomp/%05d.png

    # back to raw
    cd ../

done < "dir.txt"



