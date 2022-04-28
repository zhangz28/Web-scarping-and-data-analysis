import numpy as np
import pandas as pd
from selenium import webdriver
import time
import os
from urllib import request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

os.chdir("/Users/zhangziqi/Documents/Stats/295/")

save_path = "oc_covid"

if ~ os.path.exists(save_path):
    
    url = "https://occovid19.ochealthinfo.com/coronavirus-in-oc" # target website url
    driver = webdriver.Chrome('/Users/zhangziqi/Documents/Stats/295/chromedriver')
    driver.get(url)
    time.sleep(1)
    s = driver.find_elements_by_xpath("//*[contains(text(), 'Click to view previous weeks')]")
    s[0].click()
    s[1].click()
    content = driver.page_source.encode()

    # save webpage html into the output file
    with open(save_path, "wb") as f:
        f.write(content)

    print("Web data has been written into {0}".format(save_path))

else:

    print("Web data was written into {0}".format(save_path))

with open(save_path, "rb") as f:
    content = f.read()

soup = BeautifulSoup(content, "lxml")


# find the target
table = soup.find_all("table", {"class": "table table-striped table-responsive SchoolCaseCounts"}) # expected information is stored in these div

t0 = table[0].tbody.find_all("tr")
t1 = table[1].tbody.find_all("tr")

# store the data into dataframe
time0 = []
time1 = []

data0 = []
data1 = []

for i in range(4,len(t0)):
    row = t0[i].find_all("td")
    time0.append(row[0].get_text())
    data0.append([])
    for j in range(1,5):
        data0[-1].append(int(row[j].get_text()))
data0 = np.array(data0)

time0.reverse()
data0 = np.flipud(data0)

for i in range(4,len(t1)):
    row = t1[i].find_all("td")
    time1.append(row[0].get_text())
    data1.append([])
    for j in range(1,6):
        data1[-1].append(int(row[j].get_text()))
data1 = np.array(data1)

time1.reverse()
data1 = np.flipud(data1)


table0 = pd.DataFrame(data0)
print (table0.columns.values)
table0 = table0.rename(columns = {0: 'student', 1: 'teacher',2: 'otherstaff', 3: 'grandtotal' }) 


table1 = pd.DataFrame(data1)
print (table1.columns.values)
table1 = table1.rename(columns = {0: 'middle', 1: 'high',2: 'combined', 3: 'college', 4: "grandtotoal" }) 


np.savetxt(r'/Users/zhangziqi/Documents/Stats/295/table0.txt', table0.values, fmt='%d')
np.savetxt(r'/Users/zhangziqi/Documents/Stats/295/table1.txt', table1.values, fmt='%d')

#shape0 is number of rows, shape1 is number of columns


ind = range(table0.shape[0])
plt.figure(figsize = (20,10))
plt.plot(ind, table0["student"], label ='student' )
plt.plot(ind,table0["teacher"], label ="teacher" )
plt.plot(ind, table0["otherstaff"], label ="otherstaff" )
plt.title("Table 1")
plt.xlabel("Date")
plt.ylabel("Count")
ticks = [0, 20, 40, 60, 80]
labels = ['2020-08-16 to 2020-08-22'
, '2021-01-03 to 2021-01-09', '2021-05-23 to 2021-05-29','2021-10-10 to 2021-10-16','2022-02-27 to 2022-03-05']
plt.xticks(ticks, labels,rotation = 45)
plt.legend()


ind = range(table1.shape[0])
plt.figure(figsize = (20,10))
plt.plot(ind, table1["middle"], label ='middle' )
plt.plot(ind,table1["high"], label ="high" )
plt.plot(ind, table1["combined"], label ="combined" )
plt.plot(ind, table1["college"], label ="college" )
plt.title("Table 2")
plt.xlabel("Date")
plt.ylabel("Count")
ticks = [0, 20, 40, 60, 80]
labels = ['2020-08-16 to 2020-08-22'
, '2021-01-03 to 2021-01-09', '2021-05-23 to 2021-05-29','2021-10-10 to 2021-10-16','2022-02-27 to 2022-03-05']
plt.xticks(ticks, labels,rotation = 45)
plt.legend()



def gaussian_kernel(x):
    sigma = 0.5
    return 1/np.sqrt(2*np.pi)/sigma*np.exp(-x**2/2/sigma**2)

x_train = np.array(ind).reshape(-1,1)
x_predict = np.arange(0, table0.shape[0] - 1, 0.1).reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor as knnreg
k = 5

model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table0["student"])
y_student = model.predict(x_predict)
y_student_Chris = model.predict(np.array([[71]]))
y_student_Thks = model.predict(np.array([[67]]))
y_student_Chris
y_student_Thks

model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table0["teacher"])
y_teacher = model.predict(x_predict)
y_teacher_Chris = model.predict(np.array([[71]]))
y_teacher_Thks = model.predict(np.array([[67]]))
y_teacher_Chris
y_teacher_Thks

model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table0["otherstaff"])
y_otherstuff = model.predict(x_predict)
y_otherstuff_Chris = model.predict(np.array([[71]]))
y_otherstuff_Thks = model.predict(np.array([[67]]))
y_otherstuff_Chris
y_otherstuff_Thks


plt.figure(figsize = (20,10))
plt.plot(x_predict, y_student, label ='student' )
plt.plot(x_predict, y_teacher, label ="teacher" )
plt.plot(x_predict, y_otherstuff, label ="otherstaff" )
plt.title("Table 1")
plt.xlabel("Date")
plt.ylabel("Count")
ticks = [0, 20, 40, 60, 80]
labels = ['2020-08-16 to 2020-08-22'
, '2021-01-03 to 2021-01-09', '2021-05-23 to 2021-05-29','2021-10-10 to 2021-10-16','2022-02-27 to 2022-03-05']
plt.xticks(ticks, labels,rotation = 45)
plt.legend()


def gaussian_kernel(x):
    sigma = 0.5
    return 1/np.sqrt(2*np.pi)/sigma*np.exp(-x**2/2/sigma**2)

x_train = np.array(ind).reshape(-1,1)
x_predict = np.arange(0, table0.shape[0] - 1, 0.1).reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor as knnreg
k = 5

model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table1["middle"])
y_middle = model.predict(x_predict)
y_middle_Chris = model.predict(np.array([[71]]))
y_middle_Thks = model.predict(np.array([[67]]))
y_middle_Chris
y_middle_Thks

model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table1["high"])
y_high = model.predict(x_predict)
y_high_Chris = model.predict(np.array([[71]]))
y_high_Thks = model.predict(np.array([[67]]))
y_high_Chris
y_high_Thks


model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table1["combined"])
y_combined = model.predict(x_predict)
y_combined_Chris = model.predict(np.array([[71]]))
y_combined_Thks = model.predict(np.array([[67]]))
y_combined_Chris
y_combined_Thks


model = knnreg(n_neighbors=k, weights=gaussian_kernel)
model.fit(x_train, table1["college"])
y_college = model.predict(x_predict)
y_college_Chris = model.predict(np.array([[71]]))
y_college_Thks = model.predict(np.array([[67]]))
y_college_Chris
y_college_Thks

plt.figure(figsize = (20,10))
plt.plot(x_predict, y_middle, label ='middle' )
plt.plot(x_predict,y_high, label ="high" )
plt.plot(x_predict, y_combined, label ="combined" )
plt.plot(x_predict, y_college, label ="college" )
plt.title("Table 2")
plt.xlabel("Date")
plt.ylabel("Count")
ticks = [0, 20, 40, 60, 80]
labels = ['2020-08-16 to 2020-08-22'
, '2021-01-03 to 2021-01-09', '2021-05-23 to 2021-05-29','2021-10-10 to 2021-10-16','2022-02-27 to 2022-03-05']
plt.xticks(ticks, labels,rotation = 45)
plt.legend()
