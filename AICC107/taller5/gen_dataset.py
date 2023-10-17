import datetime
computers={}
archivo=open("unbalaced_20_80_dataset.csv","r")
filtro=open("dataset.csv","w")
header=archivo.readline().strip().split(",")
print(header,len(header))
row=[header[2],header[3],header[4],header[5],header[6],
     "day","time","hour","minute","second","seconds",header[84]]
filtro.write(",".join(row)+"\n")
for row in archivo:
    linea=row.strip().split(",")
    if linea[4] not in computers:
        computers[linea[4]]=[0,0]
    if linea[84]=="ddos":
        computers[linea[4]][1]+=1
    else:
        computers[linea[4]][0]+=1
    if linea[4]=="172.31.69.28":
        if "AM" in linea[7] or "PM" in linea[7]:
            date=datetime.datetime.strptime(linea[7],"%d/%m/%Y %I:%M:%S %p")
        else:
            date=datetime.datetime.strptime(linea[7],"%d/%m/%Y %H:%M:%S")
        year=date.year
        month=date.month
        day=date.day
        hour=str(date.hour)
        minute=str(date.minute)
        second=str(date.second)
        epoch=datetime.datetime(year, month, day,0,0,0)
        delta=date-epoch
        seconds=str(int(delta.total_seconds()))
        date=linea[7].split(" ")[0]
        time=linea[7].split(" ")[1]
        row=[linea[2],linea[3],linea[4],linea[5],linea[6],
             date,time,hour,minute,second,seconds,linea[84]]
        filtro.write(",".join(row)+"\n")
filtro.close()
archivo.close()
archivo=open("computers.csv","w")
for computer in computers.keys():
    row=[computer]
    row.extend(computers[computer])
    row=list(map(str,row))
    text=",".join(row)+"\n"
    archivo.write(text)
archivo.close()
