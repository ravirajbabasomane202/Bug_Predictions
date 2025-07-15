# Flask Bug Tracker

This is a bug tracking application built with Flask.

## Setup Instructions

### 1. Environment Variables

Create a file named `.env` in the root of the project. This file is used to store sensitive configuration and is already listed in `.gitignore`.

Copy the following into your new `.env` file:

```
# Flask Secret Keys (change these to random strings for production)
SECRET_KEY='a-very-secret-key-for-flask'
SECURITY_PASSWORD_SALT='a-very-secret-salt'

# Database URL (uncomment and set if using PostgreSQL)
# DATABASE_URL="postgresql://postgres:ravi@localhost/Mydb"

# Email Configuration (for registration emails)
# Replace 'your-email@gmail.com' with your actual Gmail address.
APP_MAIL_USERNAME='your-email@gmail.com'
APP_MAIL_PASSWORD='rrqb jwtb qyhw vyta'
```

### 2. Database Migrations

Once your environment variables are set, run the database migrations. If `DATABASE_URL` is not set, it will default to SQLite.

```bash
flask db init  # Only run this the very first time
flask db migrate -m "Initial migration"
flask db upgrade
```

### 3. Run the Application

```bash
flask run
```