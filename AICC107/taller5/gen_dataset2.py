archivo=open("dataset.csv","r")
salida=open("dataset_ts.csv","w")
header="label,date,minute,"
for i in range(60):
    if i!=59:
        header+="s"+str(i)+","
    else:
        header+="s"+str(i)+"\n"
salida.write(header)
archivo.readline()
series={}
for row in archivo:
    linea=row.strip().split(",")
    if linea[5] not in series:
        series[linea[5]]={}
    r=[linea[7],linea[8],linea[9]]
    r=list(map(int,r))
    minute=r[0]*60+r[1]
    if minute not in series[linea[5]]:
        series[linea[5]][minute]=[0]*60
    series[linea[5]][minute][int(linea[9])]+=1
for date in series.keys():
    for minute in series[date].keys():
        if(date=="20/02/2018"):
            row=["DDOS_N",date,minute]
        else:
            row=["DDOS_S",date,minute]
        row.extend(series[date][minute])
        row=list(map(str,row))
        salida.write(",".join(row)+"\n")
salida.close()
