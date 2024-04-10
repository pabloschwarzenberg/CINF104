from os import listdir
from os.path import isfile, join
from datetime import datetime

path="consumos"
consumos=[]
for filename in listdir(path):
    full_path=join(path,filename)
    if isfile(full_path):
        print(filename)
        archivo=open(full_path)
        for linea in archivo:
            linea=linea.strip().split(";")
            linea[1]=linea[1][:-5]
            date=datetime.strptime(linea[1], '%Y-%m-%d %H:%M:%S')
            year=str(date.year)
            month=str(date.month)
            day=str(date.day)
            hour=str(date.hour)
            substation=linea[5].split(" ")[0]
            row=[linea[2],substation,linea[1],linea[5]]
            consumos.append(row)
        archivo.close()
archivo=open("dataset.csv","w")
archivo.write("consumption;substation;date;node\n")
for consumo in consumos:
    archivo.write(";".join(consumo)+"\n")
archivo.close()