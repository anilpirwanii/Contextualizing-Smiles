#!/bin/bash

# Create dataset directory if it doesn't exist
mkdir -p data/smile_dataset

# Skip the header line and process each record individually
awk -F, 'NR>1 {
    url=$1
    start_time=$2
    duration=$3
    output_name=$4
    
    print "Processing URL: " url
    print "Start time: " start_time
    print "Duration: " duration
    print "Output name: " output_name
    
    cmd = "if [ -f \"data/smile_dataset/" output_name ".mp4\" ]; then "
    cmd = cmd "echo \"File data/smile_dataset/" output_name ".mp4 already exists, skipping...\"; "
    cmd = cmd "else "
    cmd = cmd "echo \"Processing: " output_name "\"; "
    cmd = cmd "yt-dlp -f \"best[ext=mp4]\" \"" url "\" -o \"temp_video.mp4\"; "
    cmd = cmd "ffmpeg -i \"temp_video.mp4\" -ss " start_time " -t " duration " -c:v libx264 -c:a aac \"data/smile_dataset/" output_name ".mp4\"; "
    cmd = cmd "rm \"temp_video.mp4\"; "
    cmd = cmd "echo \"Finished processing: " output_name "\"; "
    cmd = cmd "fi"
    
    system(cmd)
}' smile_videos.csv

echo "All videos processed!"