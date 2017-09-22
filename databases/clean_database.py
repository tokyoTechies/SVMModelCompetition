import sqlite3

conn = sqlite3.connect('accuracy_student.db')
c = conn.cursor()
c.execute('delete from accuracy_student')
conn.commit()
conn.close()
