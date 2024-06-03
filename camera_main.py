
import cv2
import numpy as np
import queue
import threading
from multiprocessing import Process, Manager, Event
import time
import os
import shutil
import glob
import tqdm
import json
import sys
from collections import deque
from enum import Enum
os.environ["PYLON_CAMEMU"] = "3"

TEST_CAMERA = True

if not TEST_CAMERA:
    from pypylon import genicam
    from pypylon import pylon


BASE_DIR = '.'
IMAGE_DIR = 'result5.28_2'
OUTPUT_DIR = 'calibrate_output5.28_2'
CAMERA_COUNT = 5
IMG_HEIGHT = 540
IMG_WIDTH = 720
CHECKERBOARD_SIZE = (4, 3)
SQUARE_SIZE = 200
START_CAMERA = 4
FPS = 5
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

class Mode(Enum):
    INTRINSIC=1
    EXTRINSIC=2
    TEST=3

# 開啟相機並將相片放進queue的函式
def test_capture_images(cameras_queue, stop):
    cap = cv2.VideoCapture(0)
    i = 0
    last_time = 0
    while not stop.is_set():
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if current_time - last_time < 1/(FPS+1)/CAMERA_COUNT:
            print("skip frame.")
            continue
        last_time = current_time
        center = (frame.shape[1]//2, frame.shape[0]//2)
        # frame = cv2.flip(frame, 1)
        frame = frame[center[1]-IMG_HEIGHT//2:center[1]+IMG_HEIGHT//2, center[0]-IMG_WIDTH//2:center[0]+IMG_WIDTH//2]

        cameras_queue[i%CAMERA_COUNT].put(frame)
        # print(f"cameras_queue size: {cameras_queue[i%CAMERA_COUNT].qsize()}")
        i += 1
        # print(f"Capture image: {i}. Camera {(i-1) % CAMERA_COUNT}: {(i-1)//CAMERA_COUNT}")

    cap.release()
    print("Stop capture_images.")

def capture_images(cameras_queue, stop):
    maxCamerasToUse = CAMERA_COUNT
    try:

        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        print(devices)
        # save_config(tlFactory, devices)

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))


        cam_cnt = cameras.GetSize()

        # Create and attach all Pylon Devices.
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            # pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)

            # Print the model name of the camera.
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        
        cameras.Open()
        for camera in cameras:
            print(camera.PixelFormat.Value)
            camera.PixelFormat.SetValue("BayerRG8")
            camera.GevIEEE1588.Value = True
            camera.SyncFreeRunTimerEnable.SetValue(True)
            camera.SyncFreeRunTimerTriggerRateAbs.SetValue(FPS)
            fps = camera.SyncFreeRunTimerTriggerRateAbs.Value
            # countOfImagesToGrab = int(fps * record_duraion)
        for camera in cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
            time.sleep(0.1)
        for camera in cameras:
            camera.SyncFreeRunTimerUpdate.Execute()
        # Starts grabbing for all cameras starting with index 0. The grabbing
        # is started for one camera after the other. That's why the images of all
        # cameras are not taken at the same time.
        # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
        # According to their default configuration, the cameras are
        # set up for free-running continuous acquisition.
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImages)

        # Grab c_countOfImagesToGrab from the cameras.
        # while True:

        # images = dict()
        # for i in range(cam_cnt):
        #     images[i] = []
        print("Recording...")
        try:
            while not stop.is_set():
            # while True:
                if not cameras.IsGrabbing():
                    break
                    pass
                grabResult = cameras.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
                # print(grabResult)

                # When the cameras in the array are created the camera context value
                # is set to the index of the camera in the array.
                # The camera context is a user settable value.
                # This value is attached to each grab result and can be used
                # to determine the camera that produced the grab result.
                cameraContextValue = grabResult.GetCameraContext()

                # Print the index and the model name of the camera.
                # print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())

                # Now, the image data can be processed.
                # print("GrabSucceeded: ", grabResult.GrabSucceeded())
                # print("SizeX: ", grabResult.GetWidth())
                # print("SizeY: ", grabResult.GetHeight())
                if grabResult.GrabSucceeded():
                    img = grabResult.GetArray()
                    image = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                    cameras_queue[cameraContextValue].put(image)
                # print(f"{cameraContextValue}: {(i >> 1):06d} {((i >> 1) / fps):.2f}")
                # cv2.imwrite(f"output/camera{cameraContextValue}/img{i>>1}.png", img)
                # cv2.imshow(f"camera{cameraContextValue}", img)
                # k = cv2.waitKey(1)
                    # print("Gray value of first pixel: ", img[0, 0])
        except KeyboardInterrupt:
            pass
        print("Record Done.")

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e)
        exitCode = 1

def find_chessboards(path, frame_id, frames, display_queue, found_count, grid, save_fps, current_camera=None, mode:Mode = Mode.TEST):
    check = 0
    chessboard_corners = None
    percent = 0.80
    grid_height, grid_width = grid.shape[:2]
    done = all(grid[:,:,4].flatten())
    # if path and path[-1] == "0":
    #     print(f"grid: {grid[:,:,4].flatten()}")
    #     print(f"done: {done}")
    frame = frames[-1] if done else frames[0]
    target_corners = np.array([[(1-percent) * frame.shape[1], (1-percent) * frame.shape[0]], [percent * frame.shape[1], (1-percent) * frame.shape[0]], [percent * frame.shape[1], percent * frame.shape[0]], [(1-percent) * frame.shape[1], percent * frame.shape[0]]], dtype=np.int32)
    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None, cv2.CALIB_CB_FAST_CHECK)
    padding = 2
    grid_width_size = (target_corners[1][0] - target_corners[0][0]) // grid_width
    grid_height_size = (target_corners[2][1] - target_corners[1][1]) // grid_height
    if grid[0, 0, 0] == 0:
        grid[:, :, 0] = np.array([[target_corners[0][0] + i * grid_width_size + padding for i in range(grid_width)] for _ in range(grid_height)])
        grid[:, :, 1] = np.array([[target_corners[0][1] + i * grid_height_size + padding for _ in range(grid_width)] for i in range(grid_height)])
        grid[:, :, 2] = np.array([[target_corners[0][0] + (i+1) * grid_width_size for i in range(grid_width)] for _ in range(grid_height)])
        grid[:, :, 3] = np.array([[target_corners[0][1] + (i+1) * grid_height_size for _ in range(grid_width)] for i in range(grid_height)])

    # 畫出 Grid
    for i in range(grid_width):
        for j in range(grid_height):
            if grid[j, i, 4] == 0:
                cv2.rectangle(frame, (grid[j, i, 0], grid[j, i, 1]), (grid[j, i, 2], grid[j, i, 3]), (100, 100, 100), 2)
            else:
                cv2.rectangle(frame, (grid[j, i, 0], grid[j, i, 1]), (grid[j, i, 2], grid[j, i, 3]), (0, 255, 0), 2)



    # 畫從中心向外80%的範圍矩形
    # cv2.rectangle(frame, tuple(target_corners[0]), tuple(target_corners[2]), (0, 255, 0), 2)
    center = (frame.shape[1]//2, int(frame.shape[0] * 0.66))
    cv2.circle(frame, center, 10, (255), 3)
    # cv2.circle(frame, tuple(target_corners[0]), 5, (0, 0, 255), -1)
    # cv2.circle(frame, tuple(target_corners[1]), 5, (0, 0, 255), -1)
    # cv2.circle(frame, tuple(target_corners[2]), 5, (0, 0, 255), -1)
    # cv2.circle(frame, tuple(target_corners[3]), 5, (0, 0, 255), -1)



    if mode == Mode.EXTRINSIC and frame_id % (FPS // save_fps) == 0:
        cv2.imwrite(f"{path}/img{frame_id//(FPS // save_fps):06d}.png", original_frame)
    
    if ret:
        # print("Found chessboard in camera {}".format(camera_idx))
        # 將找到的方格板圖像加入到 found_images 中
        chessboard_corners = np.array(corners.reshape(-1, 2), dtype=np.int32)
        chessboard_center = np.mean(chessboard_corners, axis=0, dtype=np.int32)
        # print(f"chessboard_center: {chessboard_center}")
        cv2.circle(frame, tuple(chessboard_center), 5, (0, 0, 255), -1)
        # check which grid the chessboard is in
        i, j = 0, 0
        if chessboard_center[0] - grid[0,0,0] >= 0:
            i = (chessboard_center[0] - grid[0,0,0]) // grid_width_size 
        if chessboard_center[1] - grid[0,0,1] >= 0:
            j = (chessboard_center[1] - grid[0,0,1]) // grid_height_size
        # print(f"cheesboard_center: {chessboard_center}")
        # print(f"after: {i * grid_width_size}, {j * grid_height_size}")
        # print(f"grid_width_size: {grid_width_size}, grid_height_size: {grid_height_size}")
        # print(f"i: {i}, j: {j}")

        if mode == Mode.INTRINSIC and int(path[-1]) == current_camera.value:
            if i < grid_width and j < grid_height and grid[j,i,4] == 0:
                cv2.imwrite(f"{path}/img{found_count[0]:06d}.png", original_frame)
                grid[j, i, 4] = 1
                found_count[0] += 1

        # if all_corners[0].size == 0:
        #     all_corners[0] = chessboard_corners
        # else:
        #     all_corners[0] = np.concatenate((all_corners[0], chessboard_corners), axis=0)
        # print(all_corners.size)
        cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, ret)
        # cv2.polylines(frame, [chessboard_corners], True, (0, 255, 0), 2)

    # old draw four corners
    # 找出方格板的四個角落
    # if all_corners[0].size > 0:
    #     top_left_corner = min(all_corners[0], key=lambda x: x[0] + x[1])
    #     top_right_corner = max(all_corners[0], key=lambda x: x[0] - x[1])
    #     bottom_right_corner = max(all_corners[0], key=lambda x: x[0] + x[1])
    #     bottom_left_corner = min(all_corners[0], key=lambda x: x[0] - x[1])
    #     print(f"top_left_corner: {top_left_corner}")
    #     print(f"top_right_corner: {top_right_corner}")
    #     print(f"bottom_right_corner: {bottom_right_corner}")
    #     print(f"bottom_left_corner: {bottom_left_corner}")

    #     cv2.circle(frame, tuple(top_left_corner), 5, (255, 0, 0), -1)
    #     cv2.circle(frame, tuple(top_right_corner), 5, (255, 0, 0), -1)
    #     cv2.circle(frame, tuple(bottom_right_corner), 5, (255, 0, 0), -1)
    #     cv2.circle(frame, tuple(bottom_left_corner), 5, (255, 0, 0), -1)
    #     if top_left_corner[0] < target_corners[0][0] and top_left_corner[1] < target_corners[0][1]:
    #         cv2.circle(frame, tuple(target_corners[0]), 5, (0, 255, 0), -1)
    #         check += 1
    #     if top_right_corner[0] > target_corners[1][0] and top_right_corner[1] < target_corners[1][1]:
    #         cv2.circle(frame, tuple(target_corners[1]), 5, (0, 255, 0), -1)
    #         check += 1
    #     if bottom_right_corner[0] > target_corners[2][0] and bottom_right_corner[1] > target_corners[2][1]:
    #         cv2.circle(frame, tuple(target_corners[2]), 5, (0, 255, 0), -1)
    #         check += 1
    #     if bottom_left_corner[0] < target_corners[3][0] and bottom_left_corner[1] > target_corners[3][1]:
    #         cv2.circle(frame, tuple(target_corners[3]), 5, (0, 255, 0), -1)
    #         check += 1


    

    
    if done:
        cv2.putText(frame, f"O", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("done")
        if int(path[-1]) == current_camera.value:
            current_camera.value += 1
    display_queue.put(frame)
    return done
        

def process_camera(camera_idx, image_queue, display_queue, stop, current_camera=None, mode:Mode = Mode.TEST):
    """
    mode: intrinsic, extrinsic, test
    """
    all_corners = [np.array([])]
    grid_width = 8
    grid_height = 6
    grid = np.zeros((grid_height, grid_width ,5), np.int32) # 5: x1, y1, x2, y2, is_found
    found_count = [0]
    i = 0
    if mode == Mode.INTRINSIC:
        path = f"{BASE_DIR}/{IMAGE_DIR}/calibrate{camera_idx}/camera{camera_idx}"
    elif mode == Mode.EXTRINSIC:
        path = f"{BASE_DIR}/{IMAGE_DIR}/all_calibrate/camera{camera_idx}"
    else:
        path = ""
    if path:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=False)
    frames = deque(maxlen=2)
    done = False
    while not stop.is_set():
        if not image_queue.empty():
            print(f"qsize: {image_queue.qsize()}")
            frame = image_queue.get()
            frames.append(frame)
            # print(f"Process camera {camera_idx} image: {i} start.")
            done = find_chessboards(path, i, frames, display_queue, found_count, grid, save_fps=5, current_camera=current_camera, mode=mode)
            i += 1
    print("Stop process_camera. camera_idx: ", camera_idx)

# 相片分流的函式
def split_images(image_queue, cameras_queue, stop):
    id = 0
    while not stop.is_set():
        if not image_queue.empty():
            cameras_queue[id].put(image_queue.get(), timeout=1)
            id = (id + 1) % CAMERA_COUNT
        else:
            time.sleep(1/FPS/CAMERA_COUNT/2)
    print("Stop split_images.")
# 顯示相片到視窗的函式
def display_images(display_queue, stop):
    current_fps = 0
    last_time = time.time()
    window_title = "Multi-camera Display"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    i = 0
    while not stop.is_set():
        # print(len(cameras[0]), len(cameras[1]), len(cameras[2]))
        # if all(cameras[i][idx] is not None for i in range(num_cameras)):
        check_count = 0
        # if all(not camera.empty() for camera in display_queue):
        frames = []
        for camera in display_queue:
            frame = camera.get()
            frame = np.flip(frame, axis=1)
            frames.append(frame)
        if len(frames) >= 3:
            columns = 3
            if len(frames) % columns != 0:
                padding = len(frames) % columns
                frames += [np.zeros_like(frames[0])] * padding
            rows = len(frames) // columns
            hframes = []
            for i in range(rows):
                hframes.append(np.hstack(frames[i*3:(i+1)*3]))
            frame = np.vstack(hframes)
        else:
            frame = np.hstack(frames)
        i += 1
        # print(f"Display image: {i}.")
        current_time = time.time()
        current_fps = 1 / (current_time - last_time)
        last_time = current_time
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_title, frame)
        # for i in range(num_cameras):
        #     cameras[i][idx] = None
        key = cv2.waitKey(1000 // FPS // CAMERA_COUNT // 2)
        if key == 27 or check_count == CAMERA_COUNT:  # 按下ESC鍵退出
            stop.set()
            break
    cv2.destroyAllWindows()
    print("Stop display_images.")

def compute_intrinsic(images_path:str = "", camera_dir:str = ""):
    if len(images_path) == 0:
        print(f"No images to calibrate intrinsic for {os.path.basename(camera_dir)}.")
        return False, None, None
    # Arrays to store object points and image points
    object_points = []  
    image_points = []
    output_dir = os.path.join(camera_dir, 'chessboard_images')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    for frame, image in tqdm.tqdm(enumerate(images_path)):
        # if frame % 6 != 0:
        #     continue
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        
        if ret:
            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            object_points.append(objp)
            image_points.append(corners)
            out_img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
            cv2.imwrite(f'{output_dir}/chessboard{frame}.png', out_img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (IMG_WIDTH, IMG_HEIGHT), None, None)

    return True, mtx, dist

def Undistortion(images, camera_dir, mtx, dist):
    output_dir = os.path.join(camera_dir, 'undistorted_images')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    for frame, image in tqdm.tqdm(enumerate(images)):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(f'{output_dir}/undistorted{frame}.png', dst)

def calibrate_intrinsic(manual=False):
    if manual:
        stop = Event()
        # 創建相片的queue
        image_queue = Manager().Queue()
        cameras_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
        display_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
        current_camera = Manager().Value('i', START_CAMERA)

        # 開啟相機並將相片放進queue的執行緒
        processes = []
        for i in range(CAMERA_COUNT):
            process = Process(target=process_camera, args=(i, cameras_queue[i], display_queue[i], stop, current_camera , Mode.INTRINSIC), name=f"process_camera_{i}")
            process.start()
            processes.append(process)

        display_images_p = Process(target=display_images, args=(display_queue, stop), name="display_images")
        display_images_p.start()
        processes.append(display_images_p)
        # split_images_p = Process(target=split_images, args=(image_queue, cameras_queue, stop), name="split_images")
        # split_images_p.start()
        # processes.append(split_images_p)
        if TEST_CAMERA:
            capture_images_p = Process(target=test_capture_images, args=(cameras_queue,stop), name="capture_images")
        else:
            capture_images_p = Process(target=capture_images, args=(cameras_queue,stop), name="capture_images")
        capture_images_p.start()
        processes.append(capture_images_p)

    for process in processes:
        print("Join process: ", process.name)
        if process.is_alive():
            process.join()
        print("process: ", process.name, " is alive: ", process.is_alive())
        process.close()

    

    output_dir = os.path.join(BASE_DIR, OUTPUT_DIR)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    for cam_num in range(CAMERA_COUNT):
        camera_dir = os.path.join(output_dir, f'camera{cam_num}')
        if os.path.exists(camera_dir):
            shutil.rmtree(camera_dir)
        os.makedirs(camera_dir, exist_ok=False)
        # intrinsic_images = glob.glob(f'{BASE_DIR}/{IMAGE_DIR}/calibrate{3}/camera{3}/*.png')
        intrinsic_images = sorted(glob.glob(f'{BASE_DIR}/{IMAGE_DIR}/calibrate{cam_num}/camera{cam_num}/*.png'))
        # print(intrinsic_images)
        # Compute intrinsic parameters
        ret, mtx, dist = compute_intrinsic(intrinsic_images, camera_dir)
        if not ret:
            continue
        Undistortion(intrinsic_images, camera_dir, mtx, dist)
        # draw_axis(intrinsic_images, camera_dir, mtx, dist)
        print(mtx)
        print(dist)

        # Save intrinsic parameters as JSON
        fisheye_param = {
            "class_name": "FisheyeCameraParameter",
            "convention": "opencv",
            "height": IMG_HEIGHT,
            "width": IMG_WIDTH,
            "intrinsic": mtx.tolist(),
            # "k1": 0.0,
            # "k2": 0.0,
            # "k3": 0.0,
            "k1": dist[0][0],
            "k2": dist[0][1],
            "k3": dist[0][4],
            "k4": 0.0,
            "k5": 0.0,  # Additional parameters, if you don't have them set to 0
            "k6": 0.0,
            # "p1": 0.0,
            # "p2": 0.0,
            "p1": dist[0][2],
            "p2": dist[0][3],
            "world2cam": True
        }


        # Update the fisheye_param with extrinsic parameters
        fisheye_param["extrinsic_r"] = [[0,0,0],[0,0,0],[0,0,0]]
        fisheye_param["extrinsic_t"] = [0,0,0]

        # Save combined intrinsic and extrinsic parameters
        
        print(f"Saving calibrated fisheye parameters to {camera_dir}")
        output_path = os.path.join(camera_dir, f"fisheye_param_{cam_num:02}.json")
        with open(output_path, 'w') as f:
            json.dump(fisheye_param, f, indent=4)

    print("Calibration completed.")


def calibrate_all(manual=False):
    if manual:
        stop = Event()
        # 創建相片的queue
        image_queue = Manager().Queue()
        cameras_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
        display_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]

        # 開啟相機並將相片放進queue的執行緒
        processes = []
        for i in range(CAMERA_COUNT):
            process = Process(target=process_camera, args=(i, cameras_queue[i], display_queue[i], stop, None, Mode.EXTRINSIC), name=f"process_camera_{i}")
            process.start()
            processes.append(process)

        display_images_p = Process(target=display_images, args=(display_queue, stop), name="display_images")
        display_images_p.start()
        processes.append(display_images_p)
        # split_images_p = Process(target=split_images, args=(image_queue, cameras_queue, stop), name="split_images")
        # split_images_p.start()
        # processes.append(split_images_p)
        if TEST_CAMERA:
            capture_images_p = Process(target=test_capture_images, args=(cameras_queue,stop), name="capture_images")
        else:
            capture_images_p = Process(target=capture_images, args=(cameras_queue,stop), name="capture_images")
        capture_images_p.start()
        processes.append(capture_images_p)

    for process in processes:
        print("Join process: ", process.name)
        if process.is_alive():
            process.join()
        print("process: ", process.name, " is alive: ", process.is_alive())
        process.close()

def test_camera(find_chessboard=False):
    stop = Event()  
    # 創建相片的queue
    cameras_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
    display_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]

    # 開啟相機並將相片放進queue的執行緒
    processes = []
    if find_chessboard:
        for i in range(CAMERA_COUNT):
            process = Process(target=process_camera, args=(i, cameras_queue[i], display_queue[i], stop, None, Mode.TEST), name=f"process_camera_{i}")
            process.start()
            processes.append(process)
        display_images_p = Process(target=display_images, args=(display_queue, stop), name="display_images")
    else:
        display_images_p = Process(target=display_images, args=(cameras_queue, stop), name="display_images")

    display_images_p.start()
    processes.append(display_images_p)
    # split_images_p = Process(target=split_images, args=(image_queue, cameras_queue, stop), name="split_images")
    # split_images_p.start()
    # processes.append(split_images_p)
    if TEST_CAMERA:
            capture_images_p = Process(target=test_capture_images, args=(cameras_queue,stop), name="capture_images")
    else:
        capture_images_p = Process(target=capture_images, args=(cameras_queue,stop), name="capture_images")
    capture_images_p.start()
    processes.append(capture_images_p)

    for process in processes:
        print("Join process: ", process.name)
        if process.is_alive():
            process.join()
        print("process: ", process.name, " is alive: ", process.is_alive())
        process.close()

def main():
    # calibrate_intrinsic(True)
    # calibrate_all(True)
    test_camera(True)


if __name__ == "__main__":
    main()
