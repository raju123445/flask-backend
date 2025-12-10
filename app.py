from config import db, app, jwt
from modules.auth.routes import auth_bp
from modules.assessments.routes import assess_bp

db.init_app(app)
jwt.init_app(app)

@app.route('/')
def home():
    return 'This is the home page.'

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(assess_bp, url_prefix='/assessment')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)