import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="student_managment",
    user="postgres",
    password="kill",
    host="localhost",
    # port="5432"
)

cur = conn.cursor()
print("Database connected successfully.")


# create the database if it dosen't exist
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE DATABASE student_managment ")

cur.execute(""" CREATE TABLE IF NOT EXISTS students (student_id SERIAL PRIMARY KEY ,
             name VARCHAR(50) ,
             email VARCHAR(50) ,
             phone VARCHAR(20))""")
conn.commit()
print("table created")

cur.execute("""CREATE TABLE IF NOT EXISTS courses (course_id SERIAL PRIMARY KEY ,
             course_name VARCHAR(50) ,
             credits INT)""")
conn.commit()
print("table created")

cur.execute("""CREATE TABLE IF NOT EXISTS enrollments ( enrollment_id SERIAL PRIMARY KEY , 
            student_id INT REFERENCES students(student_id),
            course_id INT REFERENCES courses (course_id), 
            grade VARCHAR(10) ) """)
conn.commit()
print("table created")

sql = "INSERT INTO students (name , email , phone) VALUES(%s,%s,%s)"
data = [ 
    ("Alice Johnson", "alice@example.com", "1234567890"),
    ("Bob Smith", "bob@example.com", "9876543210"),
    ("Charlie Brown", "charlie@example.com", "5555555555")
    ]

cur.executemany(sql, data)
conn.commit()
print("the new records was inserted")
print(cur.rowcount)

sql = "INSERT INTO courses (course_name, credits) VALUES(%s,%s)"
data = [ 
    ('Mathematics', 3),
    ('Computer Science', 4),
    ('History', 2)
    ]

cur.executemany(sql, data)
conn.commit()
print("the courses has been inserted")
print(cur.rowcount)

sql = "INSERT INTO enrollments (student_id, course_id, grade) VALUES(%s,%s,%s)"
data = [ 
    (2, 1, 'A'),
    (2, 2, 'B'),
    (2, 3, 'C'),
    
    (3, 1, 'C'),
    (3, 2, 'B'),
    (3, 3, 'A'),

    (4, 1, 'A'),
    (4, 2, 'C'),
    (4, 3, 'A')
    ]

cur.executemany(sql, data)
conn.commit()
print("the enrollments has been inserted")
print(cur.rowcount)

cur.execute("select * from students")
results = cur.fetchall()
for i in results :
    print(i)

cur.execute("select name from students where student_id in ( select student_id from enrollments where grade = 'A') ")
results = cur.fetchall()
for name in results :
    print(name)


cur.execute("select course_name,credits from courses ")
results = cur.fetchall()
for name in results :
    print(name)

cur.execute("""select s.name , c.course_name from students s 
            join enrollments e on s.student_id = e.student_id  
            join courses c on c.course_id=e.course_id
             where s.name = 'Alice Johnson' """)
results = cur.fetchall()
for name in results :
    print(name)

cur.execute("""select s.name , count(e.course_id) as total_courses from students s 
            join enrollments e on s.student_id = e.student_id  
            group by s.name """)
results = cur.fetchall()
for name in results :
    print(name)

# add a student without using enrollments
sql = "insert into students (name , email , phone) values (%s,%s,%s)"
data = ('kitt','kitt@gmail.com','36665834257')
cur.execute(sql , data)
conn.commit()
print("the new student added")

cur.execute("""select s.name from students s 
            left join enrollments e on s.student_id = e.student_id  
            where e.course_id is null """)
results = cur.fetchall()
for name in results :
    print(name)

cur.execute("select * from enrollments order by grade desc ")
results = cur.fetchall()
for name in results :
    print(name)

cur.execute(""" CREATE TABLE IF NOT EXISTS instractors (instractor_id SERIAL PRIMARY KEY ,
             instractor_name VARCHAR(50))""")
conn.commit()
print("table created")


cur.execute("""alter table enrollments 
            add column instractor_id int references instractors(instractor_id) """)
conn.commit()
print("collumn added")

sql = "INSERT INTO instractors (instractor_name, instractor_id) VALUES(%s,%s)"
data = [ 
    ('mr/mohamed', 1),
    ('mr/hassanin', 2),
    ('miss/heba', 3)
    ]

cur.executemany(sql, data)
conn.commit()
print("the instractores has been added")
print(cur.rowcount)

sql = "INSERT INTO enrollments (student_id, course_id,instractor_id, grade) VALUES(%s,%s,%s,%s)"
data = [ 
    (2, 1 , 1 , 'A'),
    (2, 2 , 2 , 'B'),
    (2, 3 , 3 , 'C'),
    
    (3, 1 , 1 , 'C'),
    (3, 2 , 2 , 'B'),
    (3, 3 , 3 , 'A'),

    (4, 1 , 1 , 'A'),
    (4, 2 , 2 , 'C'),
    (4, 3 , 3 , 'A')
    ]

cur.executemany(sql, data)
conn.commit()
print("the enrollments has been inserted")
print(cur.rowcount)


cur.execute("""
SELECT 
    c.course_name,
    i.instractor_name,
    AVG(
        CASE grade
            WHEN 'A' THEN 10
            WHEN 'B' THEN 8
            WHEN 'C' THEN 6

        END
    ) AS avg_grade
FROM enrollments e
JOIN courses c ON e.course_id = c.course_id
JOIN instractors i ON e.instractor_id = i.instractor_id
GROUP BY c.course_name, i.instractor_name
""")

results = cur.fetchall() 
for row in results: 
    print(f"Course: {row[0]}, Instructor: {row[1]}, Average Grade: {row[2]:.2f}")