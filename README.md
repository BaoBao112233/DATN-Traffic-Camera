# DATN-Traffic-Camera
1.  Check set up file:
  -  Raspberry PI 4: get_pi_requirements_pi4.sh
  -  Raspberry PI 5: set-tflite-on-rpi5.md
3.  Create folders: Video_saved, video_test
4.  Install libraries in file setup.py: pip install -r setup.txt
5.  Run file draw_shapely.py first to draw the area, the path may be the of your video or IP Camera:
  -  python3 ssd_TFLite_detect\draw_shapely.py --video_path='the path"
6.  Then run the main.py:
  -  python3 ssd_TFLite_detect\main.py --video_path='the path"
