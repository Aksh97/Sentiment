from flask import Flask, request, jsonify
import os

import chatbot.chatbot as chatbot
import graphs.graphs as graphs
import sentiments.sentiments as sentiments
import tweets.tweets as tweets
import greviences.gerv as grev



app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def predict():
    message = request.args.get('message')
    intents = chat_bot.predict_class(message, chat_bot.model)
    res = chat_bot.getResponse(intents)
    print(res)
    return jsonify(res)

@app.route("/metrics", methods=["GET","POST"])
def get_tweet_metrics():
    scope = request.args.get('scope')
    telco = request.args.get('telco')
    if scope == "day":
        return jsonify(t_graph.get_daily_metrics(telco))
    elif scope == "month":
        return jsonify(t_graph.get_monthly_metrics(telco))
    else:
        return jsonify(t_graph.get_yearly_metrics(telco))

@app.route('/userMetrics', methods=["GET"])
def get_user_metrics():
    scope = request.args.get('scope')
    telco = request.args.get('telco')

    if scope == "day":
        data = t_graph.get_user_metrics(telco)
        return jsonify(data)
    elif scope == "month":
        data = t_graph.get_user_metrics_monthly(t_graph.get_user_metrics(telco))
        return jsonify(data)
    else:
        data = t_graph.get_user_metrics(telco)
        return jsonify(data)

@app.route("/grev",methods=["GET","POST"])
def get_grev():
    telco = request.args.get('telco')
    if telco == "SINGTEL":
        data = t_grev.data
    elif telco == "STARHUB":
        data = t_grev.datas
    else:
        data = t_grev.datam
    return jsonify(data)

@app.route("/tweets",methods=["GET","POST"])
def get_tweets():
    telco = request.args.get("telco")
    data = t_tweets.get_recent_tweets(telco=telco)
    return jsonify(data)


@app.route('/tweetsData',methods=["GET","POST"])
def get_all_tweets():
    telco = request.args.get("telco")
    data = t_tweets.get_recent_tweets(500,telco=telco)
    return jsonify(data)

@app.route("/polarity", methods=["GET"])
def get_polarity():
    telco = request.args.get("telco")
    data = t_sentis.get_polarity(telco)
    return jsonify(data)

@app.route("/posCount",methods=["GET","POST"])
def get_postive_timeline():
    telco = request.args.get("telco")
    data = t_sentis.get_postive_count(telco)
    return jsonify(data)

@app.route("/negCount",methods=["GET","POST"])
def get_negative_timeline():
    telco = request.args.get("telco")
    data = t_sentis.get_negative_count(telco)
    return jsonify(data)

@app.route("/totalTweets",methods=["GET"])
def get_total_tweets():
    telco = request.args.get("telco")
    data = t_graph.get_total_tweets(telco)
    return jsonify(data)


if __name__ == "__main__":
    #Init all the ML Models
    app.config["DATA"] = os.path.join(app.root_path, 'data', 'DATA.csv')
    app.config["M1"] = os.path.join(app.root_path, 'data', 'm1.csv')
    app.config["STARHUB"] = os.path.join(app.root_path, 'data', 'starhub.csv')
    app.config["GREV"] = os.path.join(app.root_path, 'data', 'GREV.json')
    app.config["GREVS"] = os.path.join(app.root_path, 'data', 'GREVS.json')
    app.config["GREVM"] = os.path.join(app.root_path, 'data', 'GREVM.json')

    global chat_bot, t_graph, t_tweets, t_sentis, t_grev
    chat_bot = chatbot.Chatbot(app)

    t_graph = graphs.Graphs(app)
    t_tweets = tweets.Tweets(app)
    t_sentis = sentiments.Sentiments(app)
    t_grev = grev.Greviance(app)
    #Have a config file for ports and static stuff

    app.run()