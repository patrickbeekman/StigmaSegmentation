from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
from pymongo import errors
import pandas as pd
import bson
import pytz
from numpy import nan


class DataGrabber:

    def __init__(self):
        self.server = None
        try:
            self.client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=3)
            self.client.server_info()
            print("::Normal connection::")
        except errors.ServerSelectionTimeoutError:
            self.server = SSHTunnelForwarder(
                'cs.appstate.edu',
                ssh_username='bee',
                ssh_password='cs.13,bee',
                remote_bind_address=('127.0.0.1', 27017)
            )

            self.server.start()
            self.client = MongoClient('127.0.0.1', self.server.local_bind_port)
            print("::SSH connect::")
        self.db = self.client.beeDB
        self.videoDB = self.db.VideoFiles.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.weather = self.db.Weather.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.sun = self.db.SunriseSunset.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.audioDB = self.db.AudioFiles.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.tagDB = self.db.Tags.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.ground_truthDB = self.db.GroundTruths.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.hivesDB = self.db.Hives.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))
        self.hourly_videoDB = self.db.AverageTrafficByHour
        self.commentsDB = self.db.Comments.with_options(
            codec_options=bson.codec_options.CodecOptions(tz_aware=True, tzinfo=pytz.timezone('US/Eastern')))


    def shutdown(self):
        self.server.stop()

    def videos_between_dates(self, hives=[], fields=["FilePath"], start_date=None, end_date=None, set_index=True, active_bees=False):
        if len(hives) == 0 or hives is None:
            hives = self.distinct_hives()
        if start_date is None:
            start_date = self.videoDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", 1)],
                                               projection={"UTCDate": 1.0})
            start_date = start_date['UTCDate']
        if end_date is None:
            end_date = self.videoDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", -1)],
                                             projection={"UTCDate": 1.0})
            end_date = end_date['UTCDate']
        query = {}
        if active_bees:
            query['$and'] = [
                {"HiveName": {"$in": hives}},
                {"UTCDate": {"$gt": start_date}},
                {"UTCDate": {"$lt": end_date}},
                {"ArrivalsTriangle": {"$gt": 1}}
            ]
        else:
            query['$and'] = [
                {"HiveName": {"$in": hives}},
                {"UTCDate": {"$gt": start_date}},
                {"UTCDate": {"$lt": end_date}}
            ]
        # projection = {}
        # projection["FilePath"] = 1.0
        # projection['UTCDate'] = 1.0
        # projection['FileSize'] = 1.0
        projection = {}
        projection['UTCDate'] = 1.0
        for field in fields:
            projection[field] = 1.0
        sort = [("UTCDate", -1)]
        cursor = self.videoDB.find(query, projection=projection, sort=sort)
        df = pd.DataFrame(list(cursor))
        print("VID LENGTHHHHHH", len(df))
        if len(df) > 0:
            df['UTCDate'] = df['UTCDate'].dt.tz_localize(None)
        if set_index and len(df) > 0:
            df = df.set_index("UTCDate")
        return df

    def video_date_bounds(self, selected_hives=None):
        if selected_hives is None or len(selected_hives) == 0:
            selected_hives = self.hives
        start_date = self.videoDB.find_one({"HiveName": {"$in": selected_hives}}, sort=[("UTCDate", 1)],
                                           projection={"UTCDate": 1.0})
        end_date = self.videoDB.find_one({"HiveName": {"$in": selected_hives}}, sort=[("UTCDate", -1)],
                                         projection={"UTCDate": 1.0})
        return start_date['UTCDate'], end_date['UTCDate']

    def weather_date_bounds(self):
        start_date = self.weather.find_one(sort=[("UTCDate", 1)], projection={"UTCDate": 1.0})
        end_date = self.weather.find_one(sort=[("UTCDate", -1)], projection={"UTCDate": 1.0})
        return start_date['UTCDate'], end_date['UTCDate']

    def weather_between_dates(self, fields=['UTCDate'], start_date=None, end_date=None, set_index=True):
        if start_date is None:
            start_date = self.weather.find_one(sort=[("UTCDate", 1)], projection={"UTCDate": 1.0})
            start_date = start_date['UTCDate']
        if end_date is None:
            end_date = self.weather.find_one(sort=[("UTCDate", -1)], projection={"UTCDate": 1.0})
            end_date = end_date['UTCDate']

        query = {}
        query['$and'] = [
            {"UTCDate": {"$gt": start_date}},
            {"UTCDate": {"$lt": end_date}}
        ]
        projection = {}
        projection['UTCDate'] = 1.0
        for field in fields:
            projection[field] = 1.0
        sort = [("UTCDate", -1)]
        cursor = self.weather.find(query, projection=projection, sort=sort)
        df = pd.DataFrame(list(cursor))
        if len(df) != 0:
            df['UTCDate'] = df['UTCDate'].dt.tz_localize(None)
            if '2m Air Temperature (F)' in df.columns:
                try:
                    df[df['2m Air Temperature (F)'] == "Failed QC"] = nan
                except TypeError:
                    pass
            if set_index:
                df = df.set_index("UTCDate")
        print("Got Weather!!!!!", len(df))
        return df

    def distinct_hives(self):
        return self.audioDB.distinct("HiveName")

    def get_hives_date_bounds(self, specific_hives=None):
        date_bounds = {}
        if specific_hives is None:
            specific_hives = self.distinct_hives()
        for hive in specific_hives:
            start_date = self.videoDB.find_one({"HiveName": hive}, sort=[("UTCDate", 1)],
                                               projection={"UTCDate": 1.0})
            end_date = self.videoDB.find_one({"HiveName": hive}, sort=[("UTCDate", -1)],
                                             projection={"UTCDate": 1.0})
            date_bounds[hive] = (start_date['UTCDate'], end_date['UTCDate'])
        return date_bounds

    def get_all_sunrise(self):
        cursor = self.sun.find(projection={"Sunrise": 1.0})
        df = pd.DataFrame(list(cursor))
        return df

    def get_sunrise_between_dates(self, start_date, end_date):
        if start_date is None:
            start_date = self.sun.find_one(sort=[("Sunrise", 1)], projection={"Sunrise": 1.0})
            start_date = start_date['Sunrise']
        if end_date is None:
            end_date = self.sun.find_one(sort=[("Sunrise", -1)], projection={"Sunrise": 1.0})
            end_date = end_date['Sunrise']
        query = {}
        query['$and'] = [
            {"Sunrise": {"$gt": start_date}},
            {"Sunrise": {"$lt": end_date}}
        ]
        cursor = self.sun.find(query, projection={'Sunrise':1.0}, sort=[("Sunrise", -1)])
        df = pd.DataFrame(list(cursor))
        return df

    def get_all_sunset(self):
        cursor = self.sun.find(projection={"Sunset": 1.0})
        df = pd.DataFrame(list(cursor))
        return df

    def get_sunset_between_dates(self, start_date, end_date):
        if start_date is None:
            start_date = self.sun.find_one(sort=[("Sunset", 1)], projection={"Sunset": 1.0})
            start_date = start_date['Sunset']
        if end_date is None:
            end_date = self.sun.find_one(sort=[("Sunset", -1)], projection={"Sunset": 1.0})
            end_date = end_date['Sunset']
        query = {}
        query['$and'] = [
            {"Sunset": {"$gt": start_date}},
            {"Sunset": {"$lt": end_date}}
        ]
        cursor = self.sun.find(query, projection={'Sunset':1.0}, sort=[("Sunset", -1)])
        df = pd.DataFrame(list(cursor))
        return df

    def audio_between_dates(self, hives=[], fields=["FilePath"], start_date=None, end_date=None, set_index=True):
        if len(hives) == 0 or hives is None:
            hives = self.distinct_hives()
        if start_date is None:
            start_date = self.audioDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", 1)],
                                               projection={"UTCDate": 1.0})
            start_date = start_date['UTCDate']
        if end_date is None:
            end_date = self.audioDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", -1)],
                                             projection={"UTCDate": 1.0})
            end_date = end_date['UTCDate']
        query = {}
        query['$and'] = [
            {"HiveName": {"$in": hives}},
            {"UTCDate": {"$gt": start_date}},
            {"UTCDate": {"$lt": end_date}}
        ]
        projection = {}
        projection['UTCDate'] = 1.0
        for field in fields:
            projection[field] = 1.0
        sort = [("UTCDate", -1)]
        cursor = self.audioDB.find(query, projection=projection, sort=sort)
        df = pd.DataFrame(list(cursor))
        if len(df) > 0:
            df['UTCDate'] = df['UTCDate'].dt.tz_localize(None)
        if set_index and len(df) > 0:
            df = df.set_index("UTCDate")
        print("Got Audio!!!!!", len(df))
        return df

    def get_tags_from_filepaths(self, file_paths=[], set_index=False):
        query = {'FilePath': {"$in": file_paths}}
        projection = {}
        projection['UTCDate'] = 1.0
        projection['FilePath'] = 1.0
        projection['HiveName'] = 1.0
        projection['Tag'] = 1.0
        cursor = self.tagDB.find(query, projection)
        df = pd.DataFrame(list(cursor))
        if len(df) > 0:
            df['UTCDate'] = df['UTCDate'].dt.tz_localize(None)
        if set_index and len(df) > 0:
            df = df.set_index("UTCDate")
        return df

    def get_ground_truths(self):
        query = {}
        projection = {}
        projection['Annotation'] = 1.0
        projection['HiveDevice'] = 1.0
        projection['UTCStartDate'] = 1.0
        projection['UTCEndDate'] = 1.0
        projection['Notes'] = 1.0
        sort = [("UTCStartDate", -1)]
        cursor = self.ground_truthDB.find(query, projection=projection, sort=sort)
        df = pd.DataFrame(list(cursor))
        df['HiveDevice'] = self.hive_lookup_list_list(list(df['HiveDevice']))
        # df['UTCStartDate'] = df['UTCStartDate'].strftime("%m-%d-%Y %H:%M:%S")
        # df['UTCEndDate'] = df['UTCEndDate'].strftime("%m-%d-%Y %H:%M:%S")
        return df

    def hive_lookup_list_list(self, ids):
        projection = {}
        projection['HiveName'] = 1.0
        cursor = self.hivesDB.find({}, projection=projection)
        df = pd.DataFrame(list(cursor))
        out = []
        for id in ids:
            row = []
            for single in id:
                row.append(df['HiveName'][df['_id'] == bson.objectid.ObjectId(single)].values[0])
            out.append(row)
        return out

    def get_id_from_hive(self, hive):
        projection = {}
        projection['HiveName'] = 1.0
        cursor = self.hivesDB.find({}, projection=projection)
        df = pd.DataFrame(list(cursor))
        return df['_id'][df['HiveName'] == hive].values[0]

    def check_ground_truth_exists(self, id):
        query = {"_id": id}
        cursor = self.ground_truthDB.find_one(query)
        df = pd.DataFrame(list(cursor))
        return len(df) > 0

    def get_hourly_video_view(self, hives, fields, start_date=None, end_date=None):
        if len(hives) == 0 or hives is None:
            hives = self.distinct_hives()

        if start_date is None:
            start_date = self.videoDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", 1)],
                                               projection={"UTCDate": 1.0})
            start_date = start_date['UTCDate']
        if end_date is None:
            end_date = self.videoDB.find_one({"HiveName": {"$in": hives}}, sort=[("UTCDate", -1)],
                                             projection={"UTCDate": 1.0})
            end_date = end_date['UTCDate']

        query = {}
        query['$and'] = [
            {"HiveName": {"$in": hives}},
            {"UTCStartDate": {"$gt": start_date}},
            {"UTCStartDate": {"$lt": end_date}}
        ]

        projection = {}
        projection['UTCStartDate'] = 1.0
        for field in fields:
            projection[field] = 1.0
        cursor = self.hourly_videoDB.find(query, projection=projection)
        df = pd.DataFrame(list(cursor))
        return df

    def get_all_recorded_fights(self, hives=None):
        tag_query = {"Tag": "Fight"}
        tag_projection = {"FilePath": 1}
        tag_cursor = self.tagDB.find(tag_query, projection=tag_projection)
        df = pd.DataFrame(list(tag_cursor))

        if hives is None:
            com_query = {"Comment": {"$regex": '.*([fF]ight)|([aA]ttack).*'}}
        else:
            com_query = {"Comment": {"$regex": '.*([fF]ight)|([aA]ttack).*'}, "Hive": {"$in": hives}}
        com_projection = {"FilePath": 1}
        com_cursor = self.commentsDB.find(com_query, projection=com_projection)
        df = df.append(list(com_cursor))
        return df['FilePath'].unique()
