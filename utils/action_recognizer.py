import requests
import os
import cv2
import tempfile
from . import hand_detection

def upload_data(upload_file_rgb):# action recognition
    url = 'http://20.232.126.157:8083/uploadfile'
    headers = {'accept': 'application/json'}
    data = {'upload_file': open(upload_file_rgb, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response

def upload_data_grasp_release(fp_tmp_img):# grasp/release detection
    url = 'http://20.109.51.15:8083/classify_grasp_release_image'
    headers = {'accept': 'application/json'}
    data = {'upload_file': open(fp_tmp_img, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    import json
    data = json.loads(response.content)
    return data

def crop_video_and_upload(fp_mp4, hand_laterality = 'right', fp_debug=None):
    with tempfile.TemporaryDirectory() as out_dir:
        filename = fp_mp4.split('\\')[-1] 
        cap = cv2.VideoCapture(str(fp_mp4))
        fp_out_mp4_rescale = os.path.join(out_dir, filename)
        if os.path.exists(fp_out_mp4_rescale):
            return None
        # get the first frame
        ret, start_frame = cap.read()
        # get the last frame of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        ret, end_frame = cap.read()
        handloc_right, handloc_left = extract_hands(start_frame)
        if hand_laterality == 'right':
            handloc_first = handloc_right
        else:
            handloc_first = handloc_left
        handloc_right, handloc_left = extract_hands(end_frame)
        if hand_laterality == 'right':
            handloc_last = handloc_right
        else:
            handloc_last = handloc_left

        if handloc_first is None or handloc_last is None:
            return None
        # grt handloc_first with widenrate
        lefttop = (handloc_first['left'],handloc_first['top'])
        rightbottom = (handloc_first['right'],handloc_first['bottom'])
        center = (int((lefttop[0]+rightbottom[0])/2),int((lefttop[1]+rightbottom[1])/2))
        
        width = rightbottom[0] - lefttop[0]
        height = rightbottom[1] - lefttop[1]
        rate = 1.5
        lefttop_widen_first = (int(center[0]-width*rate/2),int(center[1]-height*rate/2))
        rightbottom_widen_first = (int(center[0]+width*rate/2),int(center[1]+height*rate/2))

        lefttop = (handloc_last['left'],handloc_last['top'])
        rightbottom = (handloc_last['right'],handloc_last['bottom'])
        center = (int((lefttop[0]+rightbottom[0])/2),int((lefttop[1]+rightbottom[1])/2))
        
        width = rightbottom[0] - lefttop[0]
        height = rightbottom[1] - lefttop[1]
        rate = 1.5
        lefttop_widen_last = (int(center[0]-width*rate/2),int(center[1]-height*rate/2))
        rightbottom_widen_last = (int(center[0]+width*rate/2),int(center[1]+height*rate/2))

        lefttop_widen_overlap = (max(0, min(lefttop_widen_first[0], lefttop_widen_last[0])),max(0, min(lefttop_widen_first[1], lefttop_widen_last[1])))
        rightbottom_widen_overlap = (min(end_frame.shape[1], max(rightbottom_widen_first[0], rightbottom_widen_last[0])),min(end_frame.shape[0], max(rightbottom_widen_first[1], rightbottom_widen_last[1])))
        cap.release()
        # crop the video
        cap = cv2.VideoCapture(fp_mp4)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_new = rightbottom_widen_overlap[0] - lefttop_widen_overlap[0]
        h_new = rightbottom_widen_overlap[1] - lefttop_widen_overlap[1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fp_out_mp4_rescale, fourcc, fps, (w_new, h_new))
        while True:
            ret, frame = cap.read()
            if ret:
                out.write(frame[lefttop_widen_overlap[1]:rightbottom_widen_overlap[1], lefttop_widen_overlap[0]:rightbottom_widen_overlap[0]])
                #import pdb; pdb.set_trace()
            else:
                break
        cap.release()
        out.release()
        response = upload_data(fp_out_mp4_rescale)
        if fp_debug is not None:
            # copy the video to debug_out
            import shutil
            shutil.copyfile(fp_out_mp4_rescale, fp_debug)
        #import pdb; pdb.set_trace()
        return response

def run_video_tsm_v1(fp_out_mp4, scale=None, crop=False):
    # rescale video using opencv
    if crop is True:
        file_name_orig = fp_out_mp4.split('\\')[-1]
        file_name = 'crop_' + fp_out_mp4.split('\\')[-1]
        fp_out_mp4_debug = fp_out_mp4.replace(file_name_orig, file_name)
        response = crop_video_and_upload(fp_out_mp4, fp_debug=fp_out_mp4_debug)
        if response is None:
            return ""
    else:
        if scale is not None:
            with tempfile.TemporaryDirectory() as output_dir_tmp:
                cap = cv2.VideoCapture(fp_out_mp4)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w_new = int(w * scale)
                h_new = int(h * scale)
                fp_out_mp4_rescale = os.path.join(output_dir_tmp, 'rescale.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(fp_out_mp4_rescale, fourcc, fps, (w_new, h_new))
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (w_new, h_new))
                        out.write(frame)
                    else:
                        break
                cap.release()
                out.release()
                response = upload_data(fp_out_mp4_rescale)
        else:
            fp_out_mp4_rescale = fp_out_mp4
            response = upload_data(fp_out_mp4_rescale)
    data = response.content
    import json
    data = json.loads(data)
    string_top1 = data['top_0']['label']
    sentence = ''
    for i, letter in enumerate(string_top1):
        if i and letter.isupper():
            sentence += ' '
        sentence += letter.lower()
    return sentence


def run_grasp_v1(fp_mp4, hand_laterality='right'):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        cap = cv2.VideoCapture(str(fp_mp4))
        # get the first frame
        ret, start_frame = cap.read()
        # get the last frame of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        ret, end_frame = cap.read()
        # print(output_dir_tmp+ "in run_grasp_v1")
        handloc_right, handloc_left = extract_hands(start_frame)
        if hand_laterality == 'right':
            handloc_first = handloc_right
        else:
            handloc_first = handloc_left
        handloc_right, handloc_left = extract_hands(end_frame)
        if hand_laterality == 'right':
            handloc_last = handloc_right
        else:
            handloc_last = handloc_left

        if handloc_first is None or handloc_last is None:
            return 'undetermined'
        frame_crop_first_frame = start_frame[
            handloc_first['top']:handloc_first['bottom'],
            handloc_first['left']:handloc_first['right']]
        fp_first_frame_object = os.path.join(
            output_dir_tmp, 'first_frame_hand.png')
        cv2.imwrite(fp_first_frame_object, frame_crop_first_frame)
        result_first = upload_data_grasp_release(fp_first_frame_object)
        frame_crop_last_frame = end_frame[
            handloc_last['top']:handloc_last['bottom'],
            handloc_last['left']:handloc_last['right']]
        fp_last_frame_object = os.path.join(
            output_dir_tmp, 'last_frame_hand.png')
        cv2.imwrite(fp_last_frame_object, frame_crop_last_frame)
        result_last = upload_data_grasp_release(fp_last_frame_object)
        # print(result_first)
        # print(result_last)
        if result_first['predict'] == 'Release' and \
            result_last['predict'] == 'Grasp':
            return 'grasp'
        elif result_first['predict'] == 'Grasp' and \
            result_last['predict'] == 'Release':
            return 'release'
        elif result_first['predict'] == 'Grasp' and \
            result_last['predict'] == 'Grasp':
            return 'in_manipulation'
        else:
            return 'undetermined'
def extract_hands(frame):
    json_data = hand_detection.run_image(
        frame)
    handloc_right = json_data['location_righthand']
    handloc_left = json_data['location_lefthand']
    if len(handloc_right) == 0:
        handloc_right = None
    if len(handloc_left) == 0:
        handloc_left = None
    return handloc_right, handloc_left
