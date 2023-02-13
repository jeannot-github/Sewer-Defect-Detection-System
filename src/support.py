import math

def convert_frame_to_timestamp(f, fps=25):
    total_seconds = f / fps
    hours = math.floor(total_seconds / 60 / 60)
    minutes = math.floor((total_seconds - hours*60*60) / 60)
    seconds = math.floor(total_seconds - hours*60*60 - minutes*60)

    if len(str(hours)) == 1:
        hours = '0' + str(hours)
    
    if len(str(minutes)) == 1:
        minutes = '0' + str(minutes)

    if len(str(seconds)) == 1:
        seconds = '0' + str(seconds)
    return f'{hours}:{minutes}:{seconds}'