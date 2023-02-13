import os
import torch
import yaml

# local imports
import src.layers as layers


def SDDS(PATH_TO_INSPECTION, BASE_SAVE_DIR, LOCAL, MODEL_WEIGHTS='./models/model weights.pt', DEFECT_CODES=['BAB', 'BBA', 'BAJ'], YOLO_THRESHOLD=0.53, PSNR_THRESHOLD=20):
    # load yolov5 locally or via the yolov5 Github 
    if LOCAL == True:
        model = torch.hub.load('./yolov5', 'custom', path=MODEL_WEIGHTS, source='local')
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS)

    # set inference settings
    model.conf = YOLO_THRESHOLD

    # loop through the video files of the inspection
    for video_file in os.listdir(PATH_TO_INSPECTION):
        # only read the supported video files
        if not video_file.endswith('.mpg'): 
            continue

        # set the path to the video file
        path_to_video = f'{PATH_TO_INSPECTION}/{video_file}' 

        # set the video file ID
        video_id = video_file.replace('.mpg', '') 

        # create a results folder if it does not exist already
        if not os.path.exists(BASE_SAVE_DIR):
            os.mkdir(BASE_SAVE_DIR)

        # create a seperate folder for the results of every video
        if not os.path.exists(f'{BASE_SAVE_DIR}/{video_id}'):
            os.mkdir(f'{BASE_SAVE_DIR}/{video_id}')

        # pull the video through the three SDDS layer
        df_lay1 = layers.defect_detection(DEFECT_CODES, model, path_to_video)

        # if defects were detected, select the best frame per defect
        if len(df_lay1) > 0:
            df_lay2 = layers.filter_direct_follow_up(BASE_SAVE_DIR, DEFECT_CODES, model, video_id, path_to_video, df_lay1)
            df_lay3 = layers.filter_PSNR(BASE_SAVE_DIR, DEFECT_CODES, PSNR_THRESHOLD, video_id, df_lay2)


if __name__ == "__main__":
    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)

        SDDS(PATH_TO_INSPECTION = data['PATH_TO_INSPECTION'],
             BASE_SAVE_DIR = data['BASE_SAVE_DIR'],
             LOCAL = data['LOCAL'])
