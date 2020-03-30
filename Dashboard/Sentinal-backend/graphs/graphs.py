import pandas as pd
import os
import calendar
import datetime
import json


class Graphs:
    def __init__(self,app):
        #SINGTEL
        csv_filename = os.path.join(app.config["DATA"])
        self.df = pd.read_csv(csv_filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                              '%d-%m-%Y %H.%M'))).dt.date
        self.df = self.df.set_index(['timestamp'])
        self.df.sort_index(inplace=True)

        #STARHUB
        csv_filename = os.path.join(app.config["STARHUB"])
        self.dfs = pd.read_csv(csv_filename)
        self.dfs['timestamp'] = pd.to_datetime(self.dfs['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                              '%d-%m-%Y %H.%M'))).dt.date
        self.dfs = self.dfs.set_index(['timestamp'])
        self.dfs.sort_index(inplace=True)

        #M1
        csv_filename = os.path.join(app.config["M1"])
        self.dfm = pd.read_csv(csv_filename)
        self.dfm['timestamp'] = pd.to_datetime(self.dfm['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                              '%d-%m-%Y %H.%M'))).dt.date
        self.dfm = self.dfm.set_index(['timestamp'])
        self.dfm.sort_index(inplace=True)


    def get_daily_metrics(self,telco):
        day_count = {}
        if telco == "SINGTEL":
            for row in self.df.index:
                if str(row) not in day_count.keys():
                    day_count[str(row)] = 1
                else:
                    day_count[str(row)] = day_count[str(row)] + 1
        elif telco == "STARHUB":
            for row in self.dfs.index:
                if str(row) not in day_count.keys():
                    day_count[str(row)] = 1
                else:
                    day_count[str(row)] = day_count[str(row)] + 1
        else:
            for row in self.dfm.index:
                if str(row) not in day_count.keys():
                    day_count[str(row)] = 1
                else:
                    day_count[str(row)] = day_count[str(row)] + 1

        day_count_temp = day_count
        for k in list(day_count_temp):
            if day_count[k] < 5:
                del day_count[k]

        return day_count

    def get_monthly_metrics(self,telco):
        month_count = {}
        if telco == "SINGTEL":
            for row in self.df.index:
                month = "{0} {1}".format(calendar.month_name[row.month] if isinstance(row, datetime.date) else
                                         calendar.month_name[
                                             int(row.split("-")[1])],
                                         row.year if isinstance(row, datetime.date) else int(row.split("-")[2]))
                if str(month) not in month_count.keys():
                    month_count[str(month)] = 1
                else:
                    month_count[str(month)] = month_count[str(month)] + 1
        elif telco == "STARHUB":
            for row in self.dfs.index:
                month = "{0} {1}".format(calendar.month_name[row.month] if isinstance(row, datetime.date) else
                                         calendar.month_name[
                                             int(row.split("-")[1])],
                                         row.year if isinstance(row, datetime.date) else int(row.split("-")[2]))
                if str(month) not in month_count.keys():
                    month_count[str(month)] = 1
                else:
                    month_count[str(month)] = month_count[str(month)] + 1
        else:
            for row in self.dfm.index:
                month = "{0} {1}".format(calendar.month_name[row.month] if isinstance(row, datetime.date) else
                                         calendar.month_name[
                                             int(row.split("-")[1])],
                                         row.year if isinstance(row, datetime.date) else int(row.split("-")[2]))
                if str(month) not in month_count.keys():
                    month_count[str(month)] = 1
                else:
                    month_count[str(month)] = month_count[str(month)] + 1


        date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), month_count[date_time_str]) for date_time_str in
                         list(month_count.keys())]
        date_time_obj.sort()
        lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
        nx = list(zip(*lx))
        return nx

    def get_user_metrics(self, telco):

        if telco == "SINGTEL":
            df = self.df
        elif telco == "STARHUB":
            df = self.dfs
        else:
            df = self.dfm

        # df['timestamp'] = df['timestamp'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
        # df = df.set_index('timestamp')
        daily_users = {}
        daily_users_metrics = {}
        for row in df.itertuples():
            ind = str(row.Index)
            if not (ind in daily_users.keys()):
                daily_users[ind] = [row.user_id]
                daily_users_metrics[ind] = 1
            else:
                if row.user_id in daily_users[ind]:
                    pass
                else:
                    daily_users_metrics[ind] = daily_users_metrics[ind] + 1
                daily_users[ind].append(row.user_id)
        return daily_users_metrics

    def get_user_metrics_monthly(self,daily):
        ld = pd.DataFrame(daily.items(), columns=['time', 'count'])
        ld['time'] = pd.to_datetime(ld['time'])
        ld = ld.set_index('time')
        ld.index = pd.to_datetime(ld.index)
        nx = {}
        for row in ld.itertuples():
            k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                 calendar.month_name[int(str(row.Index).split("-")[1])],
                                 row.Index.year if isinstance(row.Index, datetime.date) else
                                 int(str(row.Index).split("-")[2]))
            if k in nx.keys():
                nx[k] = nx[k] + row.count
            else:
                nx[k] = row.count
        date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), nx[date_time_str]) for
                         date_time_str in
                         list(nx.keys())]
        date_time_obj.sort()
        lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
        mx = list(zip(*lx))
        return mx


    def get_total_tweets(self, telco):
        if telco == "SINGTEL":
            data = {"total":int(self.df['text'].count())}
        elif telco == "STARHUB":
            data = {"total": int(self.dfs['text'].count())}
        else:
            data = {"total": int(self.dfm['text'].count())}
        return data

    def get_yearly_metrics(self,telco):
        year_count = {}
        if telco == "SINGTEL":
            for row in self.df.index:
                if str(row.year) not in year_count.keys():
                    year_count[str(row.year)] = 1
                else:
                    year_count[str(row.year)] = year_count[str(row.year)] + 1
        elif telco == "STARHUB":
            for row in self.dfs.index:
                if str(row.year) not in year_count.keys():
                    year_count[str(row.year)] = 1
                else:
                    year_count[str(row.year)] = year_count[str(row.year)] + 1
        else:
            for row in self.dfm.index:
                if str(row.year) not in year_count.keys():
                    year_count[str(row.year)] = 1
                else:
                    year_count[str(row.year)] = year_count[str(row.year)] + 1

        day_count_temp = year_count
        for k in list(day_count_temp):
            if year_count[k] < 5:
                del year_count[k]
        return year_count