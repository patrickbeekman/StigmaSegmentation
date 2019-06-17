import cv2
import video_selector

frame = None
video_name = "11-20-22"


def save_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/40x40_fight_testing/%s_fight(%d,%d).png" % (video_name, x, y), frame[y-20:y+20, x-20:x+20, :])
    pass


def cut_bees():
    global frame
    # vid = VideoSelector()
    # detail_name, filename, hive = vid.download_video(type='fight')
    frame_num = -1
    play_video = True
    ret = None

    cap = cv2.VideoCapture("C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/11-20-22.h264")  # + filename)
    while True:
        # locations = []
        # sub_images = None
        frame_num += 1
        if frame_num % 100 == 0:
            print("at frame %d" % frame_num)
        if play_video:
            ret, frame = cap.read()
        if ret:
            cv2.namedWindow("fightz")
            cv2.setMouseCallback("fightz", save_image)
            cv2.imshow("fightz", frame)
            key = cv2.waitKey(60)
            if key == 112:  # 'p'
                play_video = not play_video
            # elif key == 115: # 's'
            #
            # # split the image into small 40x40 windows
            # h, w, colors = frame.shape
            # im_size = 40
            # stride = 20
            # # cycle through the frame finding all (im_size X im_size) images with a stride and locations
            # for i in range(0, h - im_size, stride):
            #     for j in range(0, w - im_size, stride):
            #         if sub_images is None:
            #             sub_images = np.array([frame[i:i + im_size, j:j + im_size, :]])
            #         else:
            #             sub_images = np.concatenate((sub_images, [frame[i:i + im_size, j:j + im_size, :]]))
            #         locations.append((i, j))
            # predictions = full_model.predict(sub_images)
            # fight_predictions = np.where(np.round(predictions)[:, 1] == 1)
            # # save all predicted fights and the surrounding context
            # for idx, loc in enumerate(np.array(locations)[fight_predictions]):
            #     curr_sub = frame[loc[0]:loc[0] + 40, loc[1]:loc[1] + 40, :]
            #     # curr_sub = frame[max(0,loc[0]-40):min(loc[0]+80, h), max(0,loc[1]-40):min(loc[1]+80,w), :]
            #     # cv2.rectangle(curr_sub, (40,40), (80,80), (0,255,0), 3)
            #     detail_name = "11-20-22_"
            #     cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
            #                 + detail_name + "fight[%d](frame=%d).png" % (idx, frame_num),
            #                 curr_sub)
            #     curr_sub = frame[max(0, loc[0] - 40):min(loc[0] + 80, h), max(0, loc[1] - 40):min(loc[1] + 80, w), :]
            #     cv2.rectangle(curr_sub, (40, 40), (80, 80), (0, 255, 0), 3)
            #     cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
            #                 + detail_name + "fight[%d](frame=%d)CONTEXT.png" % (idx, frame_num),
            #                 curr_sub)
            #     # break # breakout because we have collected all sub images of a frame with fights in it
        else:
            cap.release()
            cv2.destroyAllWindows()
            break


def main():
    cut_bees()


if __name__ == "__main__":
    main()
