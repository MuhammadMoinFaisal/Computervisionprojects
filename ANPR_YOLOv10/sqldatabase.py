import sqlite3
#Connect to the SQLite database (or create it if it doesnot exist)
conn = sqlite3.connect('licensePlatesDatabase.db')

#Create a cusrsor object to interact with the datbase
cursor = conn.cursor()


#Create a table to store the License Plate Data

cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS LicensePlates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        license_plate TEXT
    )
    '''
)