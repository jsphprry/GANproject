#!/bin/bash
# $1 : the output video framerate




# counts the files in a directory
#function count(){
#	count=0
#	for i in $(ls $1); do count=$((count + 1)); done
#	echo $count
#}

# change directory to intermediate frames directory
echo -e "\ncd igis/"
cd igis/

# copy frame samples
#echo -e "\ncopy samples"
#frame_count=$(count './')
#for i in $(seq 1 $2); do cp $((($frame_count - 1) / $i)).png ../; done

# generate video
echo -e "\nffmpeg"
ffmpeg -framerate $1 -i %d.png -c:v libx264 -pix_fmt yuv420p ../igivid.mp4

# play video
echo -e "\nmpv"
mpv ../igivid.mp4
