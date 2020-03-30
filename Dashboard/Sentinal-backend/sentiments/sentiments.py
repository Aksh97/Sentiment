import pandas as pd
import os
import datetime
import calendar

class Sentiments:
    def __init__(self,app):
        self.df = pd.read_csv(os.path.join(app.config["DATA"]))
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%d-%m-%Y %H.%M'))).dt.date
        self.df = self.df.set_index(['timestamp'])

        #STARHUB
        self.dfs = pd.read_csv(os.path.join(app.config["STARHUB"]))
        self.dfs['timestamp'] = pd.to_datetime(self.dfs['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                               '%d-%m-%Y %H.%M'))).dt.date
        self.dfs = self.dfs.set_index(['timestamp'])

        #M!
        self.dfm = pd.read_csv(os.path.join(app.config["M1"]))
        self.dfm['timestamp'] = pd.to_datetime(self.dfm['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                               '%d-%m-%Y %H.%M'))).dt.date
        self.dfm = self.dfm.set_index(['timestamp'])



    def get_polarity(self,telco):
        if telco == "SINGTEL":
            return {"positive":len(self.df[(self.df['label']==1)]),"negative":len(self.df[(self.df['label']==0)])}
        elif telco == "STARHUB":
            return {"positive": len(self.dfs[(self.dfs['label'] == 1)]),
                    "negative": len(self.dfs[(self.dfs['label'] == 0)])}
        else:
            return {"positive": len(self.dfm[(self.dfm['label'] == 1)]),
                    "negative": len(self.dfm[(self.dfm['label'] == 0)])}

    def get_postive_count(self,telco):
        pos_count = {}
        if telco=="SINGTEL":
            for row in self.df[self.df['label']==1].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in pos_count.keys():
                    pos_count[k] = pos_count[k] + 1
                else:
                    pos_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), pos_count[date_time_str]) for
                             date_time_str in
                             list(pos_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx
        elif telco=="STARHUB":
            for row in self.dfs[self.dfs['label'] == 1].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in pos_count.keys():
                    pos_count[k] = pos_count[k] + 1
                else:
                    pos_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), pos_count[date_time_str]) for
                             date_time_str in
                             list(pos_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx
        else:
            for row in self.dfm[self.dfm['label'] == 1].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in pos_count.keys():
                    pos_count[k] = pos_count[k] + 1
                else:
                    pos_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), pos_count[date_time_str]) for
                             date_time_str in
                             list(pos_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx

    def get_negative_count(self,telco):
        neg_count = {}
        if telco == "SINGTEL":
            for row in self.df[self.df['label'] == 0].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in neg_count.keys():
                    neg_count[k] = neg_count[k] + 1
                else:
                    neg_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), neg_count[date_time_str]) for
                             date_time_str in
                             list(neg_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx
        elif telco=="STARHUB":
            for row in self.dfs[self.dfs['label'] == 0].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in neg_count.keys():
                    neg_count[k] = neg_count[k] + 1
                else:
                    neg_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), neg_count[date_time_str]) for
                             date_time_str in
                             list(neg_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx
        else:
            for row in self.dfm[self.dfm['label'] == 0].itertuples():
                k = "{0} {1}".format(calendar.month_name[row.Index.month] if isinstance(row.Index, datetime.date) else
                                     calendar.month_name[int(str(row.Index).split("-")[1])],
                                     row.Index.year if isinstance(row.Index, datetime.date) else
                                     int(str(row.Index).split("-")[2]))

                if k in neg_count.keys():
                    neg_count[k] = neg_count[k] + 1
                else:
                    neg_count[k] = 1
            date_time_obj = [(datetime.datetime.strptime(date_time_str, '%B %Y'), neg_count[date_time_str]) for
                             date_time_str in
                             list(neg_count.keys())]
            date_time_obj.sort()
            lx = [("{0} {1}".format(calendar.month_name[x[0].month], x[0].year), x[1]) for x in date_time_obj]
            nx = list(zip(*lx))
            return nx
