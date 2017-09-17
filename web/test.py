import sys

file = open("testfile.txt","w")
file.write(sys.argv[1])
file.close()
