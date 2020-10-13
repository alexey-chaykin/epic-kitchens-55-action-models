rm -f results.txt 

python test_video.py --video_file test_videos/put_pan2.mp4         --weight $1 --verb_ground_truth put   --noun_ground_truth pan       >> results.txt
python test_video.py --video_file test_videos/put_pot4.mp4         --weight $1 --verb_ground_truth put   --noun_ground_truth pot       >> results.txt
python test_video.py --video_file test_videos/put_pot5.mp4         --weight $1 --verb_ground_truth put   --noun_ground_truth pot       >> results.txt
python test_video.py --video_file test_videos/open_container3.mp4  --weight $1 --verb_ground_truth open  --noun_ground_truth container >> results.txt
python test_video.py --video_file test_videos/put_container4.mov   --weight $1 --verb_ground_truth put   --noun_ground_truth container >> results.txt
python test_video.py --video_file test_videos/put_container5.mov   --weight $1 --verb_ground_truth put   --noun_ground_truth container >> results.txt
python test_video.py --video_file test_videos/put_container6.mov   --weight $1 --verb_ground_truth put   --noun_ground_truth container >> results.txt
python test_video.py --video_file test_videos/close_container3.mov --weight $1 --verb_ground_truth close --noun_ground_truth container >> results.txt
python test_video.py --video_file test_videos/mix_flour5.mp4       --weight $1 --verb_ground_truth mix   --noun_ground_truth flour     >> results.txt
python test_video.py --video_file test_videos/mix_flour6.mp4       --weight $1 --verb_ground_truth mix   --noun_ground_truth flour     >> results.txt
python test_video.py --video_file test_videos/mix_flour7.mp4       --weight $1 --verb_ground_truth mix   --noun_ground_truth flour     >> results.txt
python test_video.py --video_file test_videos/put_chicken4.mp4     --weight $1 --verb_ground_truth put   --noun_ground_truth chicken   >> results.txt
python test_video.py --video_file test_videos/put_chicken5.mp4     --weight $1 --verb_ground_truth put   --noun_ground_truth chicken   >> results.txt

python plot.py --results_file results.txt --plot_title "accuracy" 
