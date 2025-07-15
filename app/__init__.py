from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from flask_login import LoginManager
from flask_heroku import Heroku

db = SQLAlchemy()  # Create db instance ONCE
mail = Mail()
login = LoginManager()

app = Flask(__name__)

load_dotenv('.env')
app.config.from_pyfile('config.py')

heroku = Heroku(app)
db.init_app(app)         # Initialize db with app
migrate = Migrate(app, db)

login.init_app(app)
login.login_view = 'login'

mail.init_app(app)
bcrypt = Bcrypt(app)

from .models import User

@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)

from app import views