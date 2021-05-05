#!/usr/bin/env python3

import serial, json, time, gzip, datetime

FLUSH_TIME = 10 #in seconds

filename = (
    f"{datetime.datetime.now().year}" \
    f"{datetime.datetime.now().month:02}"\
    f"{datetime.datetime.now().day:02}-" \
    f"{datetime.datetime.now().hour:02}"\
    f"{datetime.datetime.now().minute:02}"\
    f"{datetime.datetime.now().second:02}.json.gz"
)

with gzip.open(filename=f'/var/lib/airdata/air-{filename}', mode='at', compresslevel=3) as jsonOut:

    while True:
        try:
            message = ""
        
            uart = serial.Serial('/dev/ttyACM0', 115200, timeout=11) #uThingVOC connected over USB-CDC (Linux, RPi)
            uart.write(b'J\n')
            uart.write(b'1\n')
        except serial.SerialException:
            print("Error opening uart")
            break

        while True:
            if int(time.time()%FLUSH_TIME) == 0:
                print(f"Flushing file {str(datetime.datetime.now())}...", end="")
                jsonOut.flush()
                print(" Done!")

            try:
                message = uart.readline()
                uart.flushInput()
                message = message.decode()

                #skip wrong or incomplete readings
                try:
                    record = json.loads(message)
                    record["datetime"] = time.time()
                    json.dump(record, jsonOut, ensure_ascii=False)
                    jsonOut.write("\n")
                    #print(json.dumps(record, ensure_ascii=True))
                except Exception as e:
                    #print("Error in json or writing file!")
                    #print(e)
                    pass
            except Exception as e:
                print("Error: couldn't readline")
                print(e)
                jsonOut.close()
                exit(0)
#!/usr/bin/env python3

import serial, json, time, gzip, datetime

FLUSH_TIME = 10 #in seconds

filename = (
    f"{datetime.datetime.now().year}" \
    f"{datetime.datetime.now().month:02}"\
    f"{datetime.datetime.now().day:02}-" \
    f"{datetime.datetime.now().hour:02}"\
    f"{datetime.datetime.now().minute:02}"\
    f"{datetime.datetime.now().second:02}.json.gz"
)

with gzip.open(filename=f'/var/lib/airdata/air-{filename}', mode='at', compresslevel=3) as jsonOut:

    while True:
        try:
            message = ""
        
            uart = serial.Serial('/dev/ttyACM0', 115200, timeout=11) #uThingVOC connected over USB-CDC (Linux, RPi)
            uart.write(b'J\n')
            uart.write(b'1\n')
        except serial.SerialException:
            print("Error opening uart")
            break

        while True:
            if int(time.time()%FLUSH_TIME) == 0:
                print(f"Flushing file {str(datetime.datetime.now())}...", end="")
                jsonOut.flush()
                print(" Done!")

            try:
                message = uart.readline()
                uart.flushInput()
                message = message.decode()

                #skip wrong or incomplete readings
                try:
                    record = json.loads(message)
                    record["datetime"] = time.time()
                    json.dump(record, jsonOut, ensure_ascii=False)
                    jsonOut.write("\n")
                    #print(json.dumps(record, ensure_ascii=True))
                except Exception as e:
                    #print("Error in json or writing file!")
                    #print(e)
                    pass
            except Exception as e:
                print("Error: couldn't readline")
                print(e)
                jsonOut.close()
                exit(0)
