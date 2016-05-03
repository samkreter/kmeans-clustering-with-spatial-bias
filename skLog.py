

def skcsv(data,filename,append=1):

    #set up the append file
    if append == 1:
        f = open(filename,"a")
    else:
        f = open(filename,"w")

    f.write(','.join(map(str,data)))
    f.write('\n')

    f.close()

def skLog(status,message):
    f = open("log.txt","a")
    f.write(status+": "+message)
    f.close()

