import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import nltk as nl
import os
import json
import re


class Greviance:

    def __init__(self, app):
        self.app = app
        self.path = os.path.join(app.config["DATA"])
        self.paths = os.path.join(app.config["STARHUB"])
        self.pathm = os.path.join(app.config["M1"])
        self.prbm = {"Connectivity": {"connection", "connectivity", "network", "connect"},
                     "Internet": {"wifi", "broadband", "fibre", "internet", "5g", "speed", "4g"},
                     "Account": {"bill", "pay", "charge", "account", "customer", "detail", "payment", "subscribers",
                                 "price"},
                     "Products": {"data", "service", "mobile", "number", "tv", "sim", "retail", "insurance", "price",
                                  "offer"},
                     "Customer Support": {"singtelsupport", "team", "support"}}
        self.data = self.get_greviance("SINGTEL")
        self.datas = {
            "Account": 1546,
            "Connectivity": 982,
            "CustomerSupport": 1945,
            "Internet": 1708,
            "Products": 675
        }
        print("STARHUB")
        self.datam = {
            "Account": 140,
            "Connectivity": 70,
            "CustomerSupport": 90,
            "Internet": 190,
            "Products": 128
        }
        x = self.get_greviance("STARHUB")
        y = self.get_greviance("M1")

    def preprocess(self, telco):

        if telco == "SINGTEL":
            data = pd.read_csv(self.path, index_col=False)
        elif telco == "STARHUB":
            data = pd.read_csv(self.paths, index_col=False)
        else:
            data = pd.read_csv(self.pathm, index_col=False)
        dl = {}
        for i in range(len(data["text"])):
            temp = data["text"][i]
            if isinstance(temp, str):
                temp = nl.word_tokenize(temp)
                wnl = nl.WordNetLemmatizer()
                tokens_lem = [wnl.lemmatize(t, pos='v') for t in temp]
                text_clean = " ".join(tokens_lem)
                dl[i] = [text_clean, data['label'][i]]

        # data.to_csv("out_lemma.csv",mode='a',header=True,index=False)

        ######FOR finding classes

        for i in dl:
            temp = dl[i][0]
            print(temp, type(temp))
            temp = nl.word_tokenize(temp)
            temp = [t.lower() for t in temp]
            stop = stopwords.words('english')
            tokens_nostop = [t for t in temp if t not in stop]
            text_clean = " ".join(tokens_nostop)
            dl[i][0] = text_clean
        return dl

    def get_greviance(self, telco):
        if telco == "SINGTEL":
            if os.path.exists(self.app.config["GREV"]):
                with open(self.app.config["GREV"], 'r') as openfile:
                    # Reading from json file
                    json_object = json.load(openfile)
                return json_object
            else:
                datatest = self.preprocess(telco)
                connectivity = 0
                Internet = 0
                Account = 0
                Products = 0
                CustomerSupport = 0
                for i in datatest:
                    if datatest[i][1] == 0:
                        temp_sentence = datatest[i][0]
                        class1 = [t for t in temp_sentence.split() if t in self.prbm["Connectivity"]]
                        if len(class1) != 0:
                            connectivity += 1
                        class2 = [t for t in temp_sentence.split() if t in self.prbm["Internet"]]
                        if len(class2) != 0:
                            Internet += 1
                        class3 = [t for t in temp_sentence.split() if t in self.prbm["Account"]]
                        if len(class3) != 0:
                            Account += 1
                        class4 = [t for t in temp_sentence.split() if t in self.prbm["Products"]]
                        if len(class4) != 0:
                            Products += 1
                        class5 = [t for t in temp_sentence.split() if t in self.prbm["Customer Support"]]
                        if len(class5) != 0:
                            CustomerSupport += 1
                greviance = {"Connectivity": connectivity, "Internet": Internet, "Account": Account,
                             "Products": Products,
                             "CustomerSupport": CustomerSupport

                             }
                with open(self.app.config["GREV"], "w") as outfile:
                    json.dump(greviance, outfile)
                return greviance
        elif telco == "STARHUB":
            if os.path.exists(self.app.config["GREVS"]):
                with open(self.app.config["GREVS"], 'r') as openfile:
                    # Reading from json file
                    json_object = json.load(openfile)
                return json_object
            else:
                datatest = self.preprocess(telco)
                connectivity = 0
                Internet = 0
                Account = 0
                Products = 0
                CustomerSupport = 0
                for i in datatest:
                    if datatest[i][1] == 0:
                        temp_sentence = datatest[i][0]
                        class1 = [t for t in temp_sentence.split() if t in self.prbm["Connectivity"]]
                        if len(class1) != 0:
                            connectivity += 1
                        class2 = [t for t in temp_sentence.split() if t in self.prbm["Internet"]]
                        if len(class2) != 0:
                            Internet += 1
                        class3 = [t for t in temp_sentence.split() if t in self.prbm["Account"]]
                        if len(class3) != 0:
                            Account += 1
                        class4 = [t for t in temp_sentence.split() if t in self.prbm["Products"]]
                        if len(class4) != 0:
                            Products += 1
                        class5 = [t for t in temp_sentence.split() if t in self.prbm["Customer Support"]]
                        if len(class5) != 0:
                            CustomerSupport += 1
                greviance = {"Connectivity": connectivity, "Internet": Internet, "Account": Account,
                             "Products": Products,
                             "CustomerSupport": CustomerSupport

                             }
                with open(self.app.config["GREVS"], "w") as outfile:
                    json.dump(greviance, outfile)
                return greviance
        else:
            if os.path.exists(self.app.config["GREVM"]):
                with open(self.app.config["GREVM"], 'r') as openfile:
                    # Reading from json file
                    json_object = json.load(openfile)
                return json_object
            else:
                datatest = self.preprocess(telco)
                connectivity = 0
                Internet = 0
                Account = 0
                Products = 0
                CustomerSupport = 0
                for i in datatest:
                    if datatest[i][1] == 0:
                        temp_sentence = datatest[i][0]
                        class1 = [t for t in temp_sentence.split() if t in self.prbm["Connectivity"]]
                        if len(class1) != 0:
                            connectivity += 1
                        class2 = [t for t in temp_sentence.split() if t in self.prbm["Internet"]]
                        if len(class2) != 0:
                            Internet += 1
                        class3 = [t for t in temp_sentence.split() if t in self.prbm["Account"]]
                        if len(class3) != 0:
                            Account += 1
                        class4 = [t for t in temp_sentence.split() if t in self.prbm["Products"]]
                        if len(class4) != 0:
                            Products += 1
                        class5 = [t for t in temp_sentence.split() if t in self.prbm["Customer Support"]]
                        if len(class5) != 0:
                            CustomerSupport += 1
                greviance = {"Connectivity": connectivity, "Internet": Internet, "Account": Account,
                             "Products": Products,
                             "CustomerSupport": CustomerSupport

                             }
                with open(self.app.config["GREVM"], "w") as outfile:
                    json.dump(greviance, outfile)
                return greviance
