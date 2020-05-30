// my ffmpeg only works in base env lol

```
According to this script, the following file stucture will be generated:
    any_path_to_raw_data/0000000000.png
                         0000000001.png
                         ...
                         uncomp.mp4
                         comp.h264
                         decomp.mp4
    decomp.mp4 
    decomp              /000.png
                         001.png
                         ...

```

# .png to .mp4
ffmpeg -r 10  -start_number 0 -i %10d.png -vcodec libx264 -crf 0 -pix_fmt yuv420p -an -vf "crop=1200:360:0:0" uncomp.mp4
- -r: frame rate
- -start_number: the number where input start. 
- -i: input
- -vframes: number of frames in total
- -crf: 0(lossless)-51(shitty)
- -pix_fmt: tbh, everyone use this so I just followed
- -an: no acoustic
- -vf: video filter, to make sure our dimension can be divided by 2
    - "crop=width:height:start_x:start_y"
    - "scale=width:height"
- the last item is output file, instructions after it will be truncated (important!!!)
- b:v: specifies the target (average) bit rate for the encoder to use
    - 2M
- minrate specifies a minimum tolerance to be used
- maxrate specifies a maximum tolerance. this is only used in conjunction with bufsize
- bufsize specifies the decoder buffer size, which determines the variability of the output bitrate
    - The suggestion is to play around with the combinations of -bufsize, starting from the same value like the one specified for the -b:v option (or even half of it) and increasing it until your output bit rate starts jumping too much above/below the specified average bit rate. Then you know you've reached the limit and should lower the -bufsize value a little bit in order to get back on the safe side.

# .mp4 compressed by h264
ffmpeg -i uncomp.mp4 -an -vcodec libx264 -crf 30 comp.h264

# h264 decode
ffmpeg -framerate 10 -i comp.h264 -vf "fieldorder=bff" decomp.mp4  
- "fieldorder=bff": this is necessary so that our video won't quiver!

# video to .png frames
ffmpeg -i decomp.mp4 -framerate 10 -start_number 0 decomp/%03d.png

### We still need to call makeResidual() in utils.py to generate residual files
Both raw data ('0000000000.png') and decompressed data ('000.png') needs to be accessible to evaluate SSIM.
For example:
data/train /some_number/residual_000.png
                        ...
    /test  /some_other#/residual_000.png
                        ...
    /raw   /some_sumber/0000000000.png
                        ...
                        uncomp.mp4
           /some_other#/0000000000.png
                        ...
                        uncomp.mp4
    /decomp/some_sumber/000.png
                        ...
                        decomp.mp4
           /some_other#/000.png
                        ...
                        decomp.mp4


---

### gcp
#### scp:
gcloud compute scp --project cs231n-237623 --zone us-west1-b --recurse <local file or directory> cs348k-vm:~/
#### jupyter:
gcloud compute instances describe --project cs231n-237623 --zone us-west1-b cs348k-vm | grep googleusercontent.com | grep datalab



