// my ffmpeg only works in base env lol

# .png to .mp4
ffmpeg -r 10  -start_number 0 -i %10d.png -vframes 77 -vcodec libx264 -crf 0 -pix_fmt yuv420p -an -vf "scale=1200:360" uncomp.mp4

# .mp4 compressed by h264
ffmpeg -i uncomp.mp4 -an -vcodec libx264 -crf 30 comp.h264

# h264 decode
ffmpeg -framerate 10 -i comp.h264 -vf "fieldorder=bff" decomp.mp4  

# video to .png frames
ffmpeg -i decomp.mp4 -framerate 10 -start_number 0 decomp/%03d.png


### gcp
#### scp:
gcloud compute scp --project cs231n-237623 --zone us-west1-b --recurse <local file or directory> cs348k-vm:~/
#### jupyter:
gcloud compute instances describe --project cs231n-237623 --zone us-west1-b cs348k-vm | grep googleusercontent.com | grep datalab



