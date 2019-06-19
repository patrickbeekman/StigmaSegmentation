import cv2
import video_selector

frame = None
next_video = True
video_name = "11-20-22"
count = 0


def save_image(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/40x40_fight_testing/positive/%s_fight(%d,%d).png" % (video_name, x, y), frame[y-20:y+20, x-20:x+20, :])
        try:
            cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/80x80_fight_testing/positive/%s_fight[%d](%d,%d).png" % (video_name, count, x, y), frame[y-40:y+40, x-40:x+40, :])
        except IndexError:
            print("80x80 out of bounds error")
        count += 1
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/40x40_fight_testing/negative/%s_fight(%d,%d).png" % (video_name, x, y), frame[y-20:y+20, x-20:x+20, :])
        try:
            cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/80x80_fight_testing/negative/%s_fight[%d](%d,%d).png" % (video_name, count, x, y), frame[y-40:y+40, x-40:x+40, :])
        except IndexError:
            print("80x80 out of bounds error")
        count += 1


def cut_bees():
    global frame, video_name, next_video
    vid = video_selector.VideoSelector()
    while next_video:
        detail_name, filename, hive = vid.download_video(type='fight')
        video_name = filename[:filename.find(".")]
        frame_num = -1
        play_video = True
        ret = None

        cap = cv2.VideoCapture("C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/" + filename)
        while True:
            # locations = []
            # sub_images = None
            if play_video:
                ret, frame = cap.read()
            if ret:
                frame_num += 1
                if frame_num % 100 == 0:
                    print("at frame %d" % frame_num)
                cv2.namedWindow("fightz")
                cv2.setMouseCallback("fightz", save_image)
                cv2.imshow("fightz", frame)
                key = cv2.waitKey(60)
                if key == 32 or key == 112:  # 'space bar' or 'p'
                    play_video = not play_video
                if key == 113 or key == 115: # 'q' or 's'
                    next_video = False
            else:
                cap.release()
                cv2.destroyAllWindows()
                break
    print("Thanks 4 watching =)")

def main():
    cut_bees()


if __name__ == "__main__":
    main()
