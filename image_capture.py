import os
import sqlite3

db = sqlite3.connect("data.db")

db.execute('''CREATE TABLE CLASSES
         (CLASS_NAME           TEXT    NOT NULL,
         NO_OF_OBJECTS            INT     NOT NULL);''')

name = "unlabled"
db.execute("INSERT INTO CLASSES (CLASS_NAME,NO_OF_OBJECTS) \
      VALUES (?, 0)",(name,)); 

# def sql_fetch(con):
#     cursorObj = con.cursor()
#     cursorObj.execute('SELECT * FROM CLASSES')
#     rows = cursorObj.fetchall()
#     for row in rows:
#         print(row)

def make_class(name):
    path = "./classes/{}".format(name)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the class %s failed" % name)
    else:
        print ("Successfully created the Class %s" % name)
        db.execute("INSERT INTO CLASSES (CLASS_NAME,NO_OF_OBJECTS) \
      VALUES (?, 0)",(name,));

def add_image(frame,CLASS_NAME):
    cursor = db.execute("SELECT NO_OF_OBJECTS FROM CLASSES WHERE CLASS_NAME = ?",(CLASS_NAME,))
    for row in cursor:
        objs = row[0]
        print(objs)
    objs +=1
    db.execute("UPDATE CLASSES set NO_OF_OBJECTS = ? where CLASS_NAME = ?",(objs,CLASS_NAME)) 
    db.commit()
    img_name = "{}_{}.png".format(CLASS_NAME,objs)
    cwd = os.getcwd()
    path = os.path.join(cwd , 'classes\{}'.format(CLASS_NAME))
    print(os.path.join(path , img_name))
    cv2.imwrite(os.path.join(path , img_name), frame)
    print(img_name)              

import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        frame = increase_brightness(frame, value=50)
        add_image(frame,'unlabled')
#         print("{} written!".format(img_name))
        img_counter += 1
           
cam.release()

cv2.destroyAllWindows()