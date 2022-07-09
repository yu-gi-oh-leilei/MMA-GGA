CUDA_VISIBLE_DEVICES=0,1 python3 ./main_video_person_reid.py  --arch MMAGGA \
--train-dataset mars --test-dataset mars  --save-dir ./mars_mmagga

CUDA_VISIBLE_DEVICES=0,1 python3 ./main_video_person_reid.py  --arch MMAGGA \
--train-dataset ilidsvid --test-dataset ilidsvid  --save-dir ./ilidsvid_mmagga