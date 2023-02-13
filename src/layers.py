import pandas as pd
import torch
import os
import cv2
import shutil
import matplotlib

#local imports
import src.support as support


def defect_detection(DEFECT_CODES, model, path_to_video):
    # create a dataframe to save the results
    columns = DEFECT_CODES.copy()
    columns.append('frame')                                            
    df_results = pd.DataFrame(columns=columns)

    # load video and perform defect detection
    video = cv2.VideoCapture(path_to_video) 
    success, image = video.read()

    current_frame = 0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # perform detection
        results = model(image)
        temp_df = results.pandas().xyxy[0]
        if len(temp_df) > 0:
            temp_df = temp_df.groupby(['name']).max()

        # save the confidence for every defect class in a temporary list
        temp_result = []
        for defect_class in DEFECT_CODES:
            # if a defect of type defect_class was detected, save the confidence value
            if len(temp_df.filter(items=[defect_class], axis=0)) > 0:
                temp_result.append(temp_df[temp_df.index == defect_class]['confidence'][0])
            # else, no defect of type defect_class was detected
            else:
                temp_result.append(0)

        # if a defect is detected, thus, not all confidence values are 0, save the confidence values and frame number (e.g. [0, 0.76, 0, 5412])
        if all(x == 0 for x in temp_result) == False:
            temp_result.append(current_frame)
            df_results = pd.concat([df_results, pd.DataFrame([temp_result], columns=columns)])

        # continue to the next frame
        success, image = video.read()
        current_frame = current_frame + 1

    return df_results

def filter_direct_follow_up(BASE_SAVE_DIR, DEFECT_CODES, model, video_id, path_to_video, df):
    # select only frames with the highest confidence if a defect is identified in multiple frames in a row
    df = df.reset_index()

    # calculate the frame difference for every frame
    df['frame_difference'] = df.frame.diff()

    # create a dict with defect classes as the key, and a list of frames that should be kept as value
    to_keep = {}

    # loop through all defect classes
    for defect_class in DEFECT_CODES:
        # create an empty list per defect class
        to_keep[defect_class] = []

        # initiate the maximum confidence for the defect class at zero
        temp_max = (0,0)

        # loop through all defect detections
        for count in range(len(df)):

            #if the frame difference equals one
            if df.iloc[count, list(df.columns).index('frame_difference')] == 1:

                # if the particular defect class is detected in this frame
                if df.iloc[count, list(df.columns).index(defect_class)] > 0:
                    for i in range(1,len(df)):
                        # check if the confidence of this particular defect class in the prior frame 
                        conf = df.iloc[count-i, list(df.columns).index(defect_class)]

                        # save the count of the frame with the highest confidence (confidence is zero if defect class is different)
                        # if the confidence of count-i is greater than the currently highest confidence, set count-i as the new currently highest confidence
                        if conf > temp_max[1]:
                            temp_max = (count-i, conf)
                        # else, the currently highest confidence is greater then the confidence of count-i, thus the count-i frame should not be kept
                        else:
                            if count-i in to_keep[defect_class]:
                                to_keep[defect_class].remove(count-i)

                        # if the prior frame (count-i) was the first frame in this sequence of directly following frames        
                        if df.iloc[count-i, list(df.columns).index('frame_difference')] != 1:
                            # then keep the frame with the highest confidence 
                            to_keep[defect_class].append(temp_max[0])

                            # and start with a maximum confidence of zero again
                            temp_max = (0,0)
                            break
            # else, the frame difference does not equal one
            else:
                # if the next frame is not the last frame
                if count+1 < len(df):    
                    # if the next frame does not directly follow the current frame                                                  OVERBODIG?                       
                    if df.iloc[count+1, list(df.columns).index('frame_difference')] > 1:
                        # keep the current frame
                        to_keep[defect_class].append(count)
                # else, the next frame is the last frame
                else:
                    # keep the last frame
                    to_keep[defect_class].append(count)

    # determine the final list of frames to keep, for all defect classes combined
    final_to_keep = []
    for defect_class in DEFECT_CODES:
        final_to_keep = final_to_keep + to_keep[defect_class]

    # check for all frames with defects detected, whether they should be kept
    for count in reversed(range(len(df))):
        if count not in set(final_to_keep):
            df = df.drop(count)

    #save the selected frames with bounding boxes
    video = cv2.VideoCapture(path_to_video) 
    success, image = video.read()
    current_frame = 0
    while success:
        if current_frame in df['frame'].values:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(image)

            # save the image in the defined save directory, and name it as the frame number 
            save_dir = f'{BASE_SAVE_DIR}/{video_id}/{current_frame}'
            results.save(save_dir=save_dir)
            # the save function saves the image as "image0.jpg", which we replace with the frame number .jpg
            shutil.move(f'{save_dir}/image0.jpg', f'{BASE_SAVE_DIR}/{video_id}/{current_frame}.jpg')   
            # the save function automatically creates a new subfolder, which we remove                       
            if os.path.exists(save_dir):
                os.rmdir(save_dir)

        # Continue to the next frame
        success, image = video.read()
        current_frame = current_frame + 1

    video.release()

    return df

def filter_PSNR(BASE_SAVE_DIR, DEFECT_CODES, PSNR_THRESHOLD, video_id, df):
    columns = DEFECT_CODES.copy()                       # wss beter is --> columns = df.columns (ipv deze twee regels)
    columns.append('frame')     

    # determine which frames have been saved for this video id
    img_list = sorted([int(img.replace('.jpg', '')) for img in os.listdir(f'{BASE_SAVE_DIR}/{video_id}') if img.endswith('.jpg')])

    to_remove = []
    to_keep = []
    temp_max = (0,0)
    # loop through all frames
    for count, img in enumerate(img_list):  

        # if we already know a frame should be removed or kept, do not analyse it
        if img in to_remove or img in to_keep:
            continue
        
        # compare the frame to subsequent frames that are similar (based on PSNR)
        for i in range(1, len(df)):               
            if count+i == len(img_list):
                temp_max = (0,0) 
                break

            # compute PSNR
            similarity = cv2.PSNR(matplotlib.image.imread(f'{BASE_SAVE_DIR}/{video_id}/{img}.jpg'), matplotlib.image.imread(f'{BASE_SAVE_DIR}/{video_id}/{img_list[count+i]}.jpg'))

            # if images are similar
            if similarity > PSNR_THRESHOLD:
                max_confidence1 = max([df[df['frame'] == img][defect_code].values[0] for defect_code in DEFECT_CODES])                #WSS veranderen naar max conf per defect class
                max_confidence2 = max([df[df['frame'] == img_list[count+i]][defect_code].values[0] for defect_code in DEFECT_CODES])

                # if confidence is highest for the current image, save current image as highest confidence
                if max_confidence1 > max_confidence2:
                    to_remove.append(img_list[count+i])
                    if max_confidence1 >= temp_max[1]:
                        temp_max = (img, max_confidence1)
                    else:
                        to_remove.append(img)
                # else, confidence is highest for the image count + i, save image count + i as highest confidence
                else:
                    to_remove.append(img)
                    if max_confidence2 >= temp_max[1]:
                        temp_max = (img_list[count+i], max_confidence2)
                    else:
                        to_remove.append(img_list[count+i])
            
            # else, images are dissimilar
            else:
                to_keep.append(temp_max[0])
                temp_max = (0,0)
                break

    # remove the images that we consider to be duplicates with non-optimal confidence
    for file in set(to_remove):
        os.remove(f'{BASE_SAVE_DIR}/{video_id}/{file}.jpg')

    # save a csv that contains the frame number (same as filename image), the timestamp in the video, and the defect code
    # determine which images are saved
    df_save = df.copy()[columns]
    df_save = df_save[~df_save['frame'].isin(to_remove)]

    # present the defect code instead of confidence value
    for defect_code in DEFECT_CODES:
        df_save.loc[df_save[defect_code] > 0, defect_code] = defect_code

    # transform the frame number into the timestamp
    df_save['timestamp'] = df_save.apply(lambda x: support.convert_frame_to_timestamp(x['frame']), axis=1)

    # make the csv more readable by changing the order of the columns
    col = df_save.pop("frame")
    df_save.insert(0, col.name, col)

    col = df_save.pop("timestamp")
    df_save.insert(1, col.name, col)

    # save the csv
    df_save.to_csv(f'{BASE_SAVE_DIR}/{video_id}/{video_id}.csv', index=False)

    return df_save