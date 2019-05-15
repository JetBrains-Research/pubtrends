from bokeh.embed import components
from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap

from keypaper.analysis import KeyPaperAnalyzer
from keypaper.visualization import Plotter

app = Flask(__name__)
Bootstrap(app)
# app.config['BOOTSTRAP_SERVE_LOCAL'] = True

FORM_HTML = '''<form method="POST">
                  Enter search terms:<br>
                  <input type="text" name="terms"><input type="submit" value="Submit">
               </form>'''

# Index page, no args
@app.route('/', methods=['GET', 'POST'])
def index():
    if len(request.args) > 0:
        terms = request.args.get('terms').split('+')
        analyzer.launch(*terms)
        data = []
        for p in plotter.subtopic_timeline_graphs():
            data.append(components(p))
        return render_template('results.html', search_string=' '.join(terms), data=data)

    if request.method == 'POST':
        terms = request.form.get('terms').split(' ')
        redirect_url = '+'.join(terms)
        return redirect(f'/?terms={redirect_url}')

    return render_template('main.html')

# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
    analyzer = KeyPaperAnalyzer()
    plotter = Plotter(analyzer)
    app.run(port=5000, debug=True, extra_files=['templates/'])