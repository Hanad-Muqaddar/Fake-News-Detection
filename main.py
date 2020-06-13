from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvectriozer = TfidfVectorizer(stop_words='english', max_df=0.7)

file = open('model.pkl', 'rb')
pac = pickle.load(file)
file.close()

@app.route('/')
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        news = str(mydict['news'])
        input = [news]
        check= tfidfvectriozer.transform(input)
        result = pac.predict(check)

        return render_template('show.html', inf=result)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)