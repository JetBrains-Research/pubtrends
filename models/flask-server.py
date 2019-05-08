# Hello, Flask!
from flask import Flask, render_template, request, redirect

from keypaper.analysis import KeyPaperAnalyzer

app = Flask(__name__)
analyzer = KeyPaperAnalyzer('nikolay.kapralov@gmail.com')

FORM_HTML = '''<form method="POST">
                  Enter search terms:<br>
                  <input type="text" name="terms"><input type="submit" value="Submit">
               </form>'''

# Index page, no args
@app.route('/', methods=['GET', 'POST'])
def index():
    if len(request.args) > 0:
        terms = request.args.get('terms').split('+')
        analyzer.search(*terms)
        analyzer.load_publications()
        return f"Found {len(analyzer.pub_df)} publications:<br>{analyzer.pub_df.to_html()}"

    if request.method == 'POST':
        terms = request.form.get('terms').split(' ')
        redirect_url = '+'.join(terms)
        return redirect(f'/?terms={redirect_url}')
    
    return FORM_HTML

# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
    app.run(port=5000, debug=True)