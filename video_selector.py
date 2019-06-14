import DataGrabber
import pickle
from datetime import datetime
import subprocess
import numpy as np
import pandas as pd


class VideoSelector:

    def __init__(self):
        self.dg = DataGrabber.DataGrabber()
        try:
            infile = open("selected_videos.pkl", 'rb')
            try:
                self.selected_videos = pickle.load(infile)
            except EOFError:
                self.selected_videos = []
            infile.close()
        except FileNotFoundError:
            infile = open("selected_videos.pkl", 'w+')
            infile.close()
            self.selected_videos = []
        try:
            self.selected_fight_videos = np.array(pd.read_csv('selected_fight_videos.csv', header=None)).reshape(-1)
        except FileNotFoundError:
            infile2 = open("selected_fight_videos.csv", 'w+')
            infile2.close()
            self.selected_fight_videos = []
        except pd.errors.EmptyDataError:
            infile2 = open("selected_fight_videos.csv", 'w+')
            infile2.close()
            self.selected_fight_videos = []
        try:
            self.selected_robbery_videos = np.array(pd.read_csv('selected_robbery_videos.csv', header=None)).reshape(-1)
        except FileNotFoundError:
            infile2 = open("selected_robbery_videos.csv", 'w+')
            infile2.close()
            self.selected_robbery_videos = []
        except pd.errors.EmptyDataError:
            infile2 = open("selected_robbery_videos.csv", 'w+')
            infile2.close()
            self.selected_robbery_videos = []

    def pick_video_from_set(self, hive):
        print("Picking an unseen video")
        df = self.dg.videos_between_dates(hives=[hive], fields=["FilePath"], start_date=datetime(year=2018, month=1, day=1),
                                     end_date=datetime(year=2018, month=12, day=30), set_index=False)
        single_vid = df.sample(n=1).iloc[0]['FilePath']
        orig_len = len(self.selected_videos)
        while len(self.selected_videos) == orig_len:
            if single_vid not in self.selected_videos:
                self.selected_videos.append(single_vid)
            else:
                single_vid = df.sample(n=1).iloc[0]['FilePath']
        # write to file again and return
        outfile = open("selected_videos.pkl", 'wb')
        pickle.dump(self.selected_videos, outfile)
        outfile.close()
        print("Found:  %s" % single_vid)
        return single_vid

    def pick_unseen_fight(self):
        print("Picking unseen fight video")
        df = self.dg.get_all_recorded_fights(hives=['rpi11b', 'rpi12b', 'rpi24'])
        unseen = np.array(list(set(df).difference(set(self.selected_fight_videos))))
        if len(unseen) == 0:
            print("No more labeled fight videos in the database...")
            exit(0)
        # randomly pick a video from the set of unseen
        single_vid = np.random.choice(unseen, 1)[0]
        self.selected_fight_videos = np.append(self.selected_fight_videos, single_vid)
        # write to file again and return
        pd.DataFrame(self.selected_fight_videos).to_csv("selected_fight_videos.csv", header=None, index=None)
        print("Found:  %s" % single_vid)
        return single_vid

    def pick_robbery(self):
        print("Picking unseen robbery video")
        df = self.dg.videos_between_dates(hives=['rpi11b', 'rpi12b', 'rpi24'], start_date=datetime(year=2018, month=9, day=27), end_date=datetime(year=2018, month=11, day=1), active_bees=True)
        unseen = np.array(list(set(df['FilePath']).difference(set(self.selected_robbery_videos))))
        # randomly pick a video from the set of unseen
        single_vid = np.random.choice(unseen, 1)[0]
        self.selected_robbery_videos = np.append(self.selected_robbery_videos, single_vid)
        # write to file again and return
        pd.DataFrame(self.selected_robbery_videos).to_csv("selected_robbery_videos.csv", header=None, index=None)
        print("Found:  %s" % single_vid)
        return single_vid

    '''
        Downloads bee videos from the cs server.
        :param type - choose from one of three options [all, fight, robbery] or give it a full video path
    '''
    def download_video(self, type="all", hive="", user='windows'):
        if type == "all":
            vid_path = self.pick_video_from_set(hive)
        elif type == "fight":
            vid_path = self.pick_unseen_fight()
        elif type == "robbery":
            vid_path = self.pick_robbery()
        else:
            vid_path = type
        print("Downloading video...")
        # need to use putty cmd line
        if user == 'windows':
            # https://stackoverflow.com/questions/32598361/python-script-for-ssh-through-putty
            subprocess.check_output("pscp -pw cs.13,bee bee@cs.appstate.edu:%s C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/" % vid_path, shell=True)
        elif user == 'linux':
            subprocess.check_output("sshpass -p cs.13,bee scp -r bee@cs.appstate.edu:%s C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/" % vid_path, shell=True)
        print("Video Download complete")
        return vid_path.split("/usr/local/bee/beemon/")[1].replace('video/', '').replace('/', '@')[:-5], vid_path.split("/")[-1], vid_path.split("/")[5]
