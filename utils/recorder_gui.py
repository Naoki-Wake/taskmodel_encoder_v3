import PySimpleGUI as sg
import os
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord
import cv2
from typing import Optional, Tuple
import numpy as np
import time


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def get_file_info():
    layout = [
        [
            sg.Text(
                'Saving file name', size=(
                    40, 1), justification='center', font='Helvetica 20')], [
            sg.FolderBrowse(
                font='Helvetica 14'), sg.Text(
                "Folder name", font='Helvetica 14'), sg.InputText(
                font='Helvetica 14')], [
            sg.Text(
                "File name", font='Helvetica 14'), sg.InputText(
                font='Helvetica 14')], [
            sg.Submit(
                key="submit", font='Helvetica 14'), sg.Cancel(
                "Exit", font='Helvetica 14')]]

    window = sg.Window("file selection", layout, location=(800, 400))

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values[0] == "" or values[1] == "":
                sg.popup("Enter file information")
                continue
            else:
                fp_dir = values[0]
                fp_base = values[1]
                break
    window.close()
    return fp_dir, fp_base


def ui(fp_dir, fp_base, preview_fps=5, extraction=False):
    if extraction:
        color_frame_count = 0
        depth_frame_count = 0
        fs = 30
        W, H = 1280, 720
        depth_stack = []
        fp_out_mp4 = os.path.join(fp_dir, fp_base + ".mp4")
        fp_out_depth_mp4 = os.path.join(fp_dir, fp_base + "_depth.mp4")
        fp_out_depth_npy = os.path.join(fp_dir, fp_base + "_depth.npy")
        videowriter = cv2.VideoWriter(
            fp_out_mp4, cv2.VideoWriter_fourcc(
                *"mp4v"), fs, (W, H))
        videowriter_d = cv2.VideoWriter(
            fp_out_depth_mp4, cv2.VideoWriter_fourcc(
                *"mp4v"), fs, (W, H))

        # sg.theme('Black')
    deviceid = 0
    imageformat = ImageFormat.COLOR_MJPG
    print(f"Starting device #{deviceid}")
    config = Config(color_format=imageformat)
    device = PyK4A(config=config, device_id=deviceid)
    device.start()
    fp_mkv = os.path.join(fp_dir, fp_base + ".mkv")
    print(f"Open record file {fp_mkv}")
    record = PyK4ARecord(device=device, config=config, path=fp_mkv)
    record.create()

    import pyaudio
    import wave
    import numpy as np
    p = pyaudio.PyAudio()
    # Find out the index of Azure Kinect Microphone Array
    azure_kinect_device_name = "Azure Kinect Microphone Array"
    index = -1
    for i in range(p.get_device_count()):
        # print(p.get_device_info_by_index(i))
        if azure_kinect_device_name in p.get_device_info_by_index(
                i)["name"] and p.get_device_info_by_index(i)["hostApi"] == 3:
            index = i
            break
    if index == -1:
        print("Could not find Azure Kinect Microphone Array. Make sure it is properly connected.")
        exit()
    input_format = pyaudio.paInt32
    input_sample_width = 4
    input_channels = 7

    input_sample_rate = 48000
    stream = p.open(
        format=input_format,
        channels=input_channels,
        rate=input_sample_rate,
        input=True,
        input_device_index=index)
    # Read frames from microphone and write to wav file
    fp_audio = os.path.join(fp_dir, fp_base + ".wav")
    # define the window layout
    layout = [
        [
            sg.Text(
                'Recorder', size=(
                    40, 1), justification='center', font='Helvetica 20')], [
            sg.Image(
                filename='', key='image')], [
            sg.Button(
                'Record', size=(
                    10, 1), font='Helvetica 14'), sg.Button(
                'Stop', size=(
                    10, 1), font='Helvetica 14'), sg.Button(
                'Exit', size=(
                    10, 1), font='Helvetica 14'), ]]
    # create the window and show it without the plot
    window = sg.Window('Recorder: Azure Kinect',
                       layout, location=(800, 400))
    recording = False
    # get current time in ms
    past_time = int(round(time.time() * 1000))

    with wave.open(fp_audio, "wb") as outfile:
        # We want to write only first channel from each frame
        outfile.setnchannels(1)
        outfile.setsampwidth(input_sample_width)
        outfile.setframerate(input_sample_rate)
        while True:
            event, values = window.read(timeout=20)
            if event == 'Exit' or event == sg.WIN_CLOSED:
                break

            elif event == 'Record':
                recording = True
                window['Record'].update(disabled=True)
            elif event == 'Stop':
                recording = False
                # img = np.full((480, 640), 255)
                # # this is faster, shorter and needs less includes
                # imgbytes = cv2.imencode('.png', img)[1].tobytes()
                # window['image'].update(data=imgbytes)
                record.flush()
                record.close()
                break
            if recording:
                # ret, frame = cap.read()
                # imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
                # window['image'].update(data=imgbytes)
                # return
                capture = device.get_capture()
                record.write_capture(capture)

                if preview_fps > 0:
                    # get current time in ms
                    current_time = int(round(time.time() * 1000))
                    # calculate the time difference between current time and
                    # past time
                    time_diff = current_time - past_time
                    # if time difference is greater than 1000 ms, we are
                    # displaying frames per second
                    if time_diff > int(1000 * (1.0 / preview_fps)):
                        past_time = current_time
                        if capture.color is not None:
                            frame = convert_to_bgra_if_required(
                                imageformat, capture.color)
                            scale = 0.3
                            W, H = frame.shape[1], frame.shape[0]
                            resized = cv2.resize(
                                frame, (int(W * scale), int(H * scale)),
                                interpolation=cv2.cv2.INTER_NEAREST)
                            cv2.putText(resized,
                                        str("Preview: {0:.1f} FPS".format(1000.0 / time_diff)),
                                        (15, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (240, 230, 0),
                                        1,
                                        cv2.LINE_AA)
                            imgbytes = cv2.imencode('.png', resized)[
                                1].tobytes()  # ditto
                            window['image'].update(data=imgbytes)
                if extraction:
                    if capture.color is not None:
                        frame = convert_to_bgra_if_required(
                            imageformat, capture.color)
                        videowriter.write(frame)
                        color_frame_count += 1
                    if capture.transformed_depth is not None:
                        frame_d = colorize(
                            capture.transformed_depth, (None, 5000))
                        depth_stack.append(capture.transformed_depth)
                        videowriter_d.write(frame_d)
                        depth_frame_count += 1

                        scale = 0.3
                        resized = cv2.resize(
                            frame, (int(W * scale), int(H * scale)),
                            interpolation=cv2.cv2.INTER_NEAREST)
                        imgbytes = cv2.imencode('.png', resized)[
                            1].tobytes()  # ditto
                        window['image'].update(data=imgbytes)

                available_frames = stream.get_read_available()
                read_frames = stream.read(available_frames)
                first_channel_data = np.frombuffer(
                    read_frames, dtype=np.int32)[0::7].tobytes()
                outfile.writeframesraw(first_channel_data)
            else:
                current_time = int(round(time.time() * 1000))
                # calculate the time difference between current time and
                # past time
                time_diff = current_time - past_time
                capture = device.get_capture()
                if capture.color is not None:
                    past_time = current_time
                    frame = convert_to_bgra_if_required(
                        imageformat, capture.color)
                    scale = 0.3
                    W, H = frame.shape[1], frame.shape[0]
                    resized = cv2.resize(
                        frame, (int(W * scale), int(H * scale)),
                        interpolation=cv2.cv2.INTER_NEAREST)
                    cv2.putText(resized,
                                str("Preview: {0:.1f} FPS".format(1000.0 / time_diff)),
                                (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (240, 230, 0),
                                1,
                                cv2.LINE_AA)
                    imgbytes = cv2.imencode('.png', resized)[
                        1].tobytes()  # ditto
                    window['image'].update(data=imgbytes)
    window.close()
    print(f"{record.captures_count} frames written.")
    device.stop()
    stream.stop_stream()
    stream.close()
    p.terminate()
    if extraction:
        videowriter.release()
        videowriter_d.release()
        depth_data = np.stack(depth_stack)
        np.save(fp_out_depth_npy, depth_data)
    return fp_mkv, fp_audio


def run():
    fp_dir, fp_base = get_file_info()
    fp_mkv, fp_audio = ui(fp_dir, fp_base, preview_fps=5, extraction=False)
    return fp_mkv, fp_audio, fp_dir


if __name__ == '__main__':
    fp_dir, fp_base = get_file_info()
    ui(fp_dir, fp_base)
