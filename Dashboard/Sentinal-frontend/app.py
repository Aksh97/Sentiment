from flask import Flask,jsonify, render_template, redirect, request, url_for
import requests


app = Flask(__name__)


@app.route('/chatbot')
def chatbot():
    return render_template('tabs/chatbot.html')

@app.route('/chat',methods=["POST"])
def chat():
    message = request.form['message']
    querystring = {"message": message}
    reply = requests.request("POST",app.config["CHAT"],params=querystring).json()

    return reply


@app.route('/timeline')
def timeline():

    pos_data_sing = requests.get(app.config["POS"],params={"telco":"SINGTEL"}).json()
    neg_data_sing = requests.get(app.config["NEG"],params={"telco":"SINGTEL"}).json()

    pos_data_star = requests.get(app.config["POS"],params={"telco":"STARHUB"}).json()
    neg_data_star = requests.get(app.config["NEG"],params={"telco":"STARHUB"}).json()

    pos_data_m = requests.get(app.config["POS"],params={"telco":"M1"}).json()
    neg_data_m = requests.get(app.config["NEG"],params={"telco":"M1"}).json()

    return render_template('tabs/timeline.html',pos_data=pos_data_sing,neg_data=neg_data_sing,
                           pos_data_star=pos_data_star,neg_data_star=neg_data_star,pos_data_m=pos_data_m,neg_data_m=neg_data_m)

@app.route('/tweets')
def tweets():
    t_data = requests.get(app.config["TWEETS_DATA"]).json()
    return render_template('tabs/tweets.html',t_data=t_data)

@app.route('/m1')
def m1():
    try:
        #timeline graph
        metrics_d = requests.get(app.config["METRICS"],params={"scope": "day","telco":"M1"}).json()
        metrics_m = requests.get(app.config["METRICS"],params={"scope": "month","telco":"M1"}).json()
        metrics_y = requests.get(app.config["METRICS"], params={"scope": "year", "telco": "M1"}).json()

        #Most recent tweets
        t_data = requests.get(app.config["TWEETS"],params={"telco":"M1"}).json()

        #Sentiment data
        s_data = requests.get(app.config["POLARITY"],params={"telco":"M1"}).json()

        #Unique users
        user_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"day","telco":"M1"}).json()
        user__monthly_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"month",
                                                                                "telco":"M1"}).json()

        #greviences
        g_data = requests.get(app.config["GREV"],params={"telco":"M1"}).json()

        #Total data
        total_tweets = requests.get(app.config["TOTAL_TWEETS"],params={"telco":"M1"}).json()


        return render_template('index.html',t_data=t_data, s_data=s_data,metrics_d=metrics_d,metrics_m=metrics_m,
                               metrics_y=metrics_y,
                               user_metrics=user_metrics,total_tweets=total_tweets,
                               user__monthly_metrics=user__monthly_metrics,g_data=g_data)
    except Exception as m:
        print(m)
        return render_template('errors/500.html')


@app.route('/starhub')
def starhub():
    try:
        #timeline graph
        metrics_d = requests.get(app.config["METRICS"],params={"scope": "day","telco":"STARHUB"}).json()
        metrics_m = requests.get(app.config["METRICS"],params={"scope": "month","telco":"STARHUB"}).json()
        metrics_y = requests.get(app.config["METRICS"], params={"scope": "year", "telco": "STARHUB"}).json()

        #Most recent tweets
        t_data = requests.get(app.config["TWEETS"],params={"telco":"STARHUB"}).json()

        #Sentiment data
        s_data = requests.get(app.config["POLARITY"],params={"telco":"STARHUB"}).json()

        #Unique users
        user_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"day","telco":"STARHUB"}).json()
        user__monthly_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"month",
                                                                                "telco":"STARHUB"}).json()

        #greviences
        g_data = requests.get(app.config["GREV"],params={"telco":"STARHUB"}).json()

        #Total data
        total_tweets = requests.get(app.config["TOTAL_TWEETS"],params={"telco":"STARHUB"}).json()


        return render_template('index.html',t_data=t_data, s_data=s_data,metrics_d=metrics_d,metrics_m=metrics_m,metrics_y=metrics_y,
                               user_metrics=user_metrics,total_tweets=total_tweets,
                               user__monthly_metrics=user__monthly_metrics,g_data=g_data)
    except Exception as m:
        print(m)
        return render_template('errors/500.html')

@app.route('/')
def index():
    try:
        #timeline graph
        metrics_d = requests.get(app.config["METRICS"],params={"scope": "day","telco":"SINGTEL"}).json()
        metrics_m = requests.get(app.config["METRICS"],params={"scope": "month","telco":"SINGTEL"}).json()
        metrics_y = requests.get(app.config["METRICS"], params={"scope": "year", "telco": "SINGTEL"}).json()


        #Most recent tweets
        t_data = requests.get(app.config["TWEETS"],params={"telco":"SINGTEL"}).json()

        #Sentiment data
        s_data = requests.get(app.config["POLARITY"],params={"telco":"SINGTEL"}).json()

        #Unique users
        user_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"day","telco":"SINGTEL"}).json()
        user__monthly_metrics = requests.get(app.config["USER_METRICS"],params={"scope":"month","telco":"SINGTEL"}).json()

        #greviences
        g_data = requests.get(app.config["GREV"],params={"telco":"SINGTEL"}).json()

        #Total data
        total_tweets = requests.get(app.config["TOTAL_TWEETS"],params={"telco":"SINGTEL"}).json()


        return render_template('index.html',t_data=t_data, s_data=s_data,metrics_d=metrics_d,metrics_m=metrics_m,
                               metrics_y=metrics_y,
                               user_metrics=user_metrics,total_tweets=total_tweets,
                               user__monthly_metrics=user__monthly_metrics,g_data=g_data)
    except Exception as m:
        print(m)
        return render_template('errors/500.html')


if __name__ == '__main__':
    app.config.update(
        TWEETS_DATA="http://127.0.0.1:5000/tweetsData",
        TOTAL_TWEETS="http://127.0.0.1:5000/totalTweets",
        GREV = "http://127.0.0.1:5000/grev",
        USER_METRICS = "http://127.0.0.1:5000/userMetrics",
        POLARITY = "http://127.0.0.1:5000/polarity",
        TWEETS = "http://127.0.0.1:5000/tweets",
        METRICS = "http://127.0.0.1:5000/metrics",
        CHAT = "http://127.0.0.1:5000/chat",
        POS = "http://127.0.0.1:5000/posCount",
        NEG = "http://127.0.0.1:5000/negCount"
    )
    app.run(port=9001)
