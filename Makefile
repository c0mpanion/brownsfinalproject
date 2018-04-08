OBJECTS = main.o scoring.o

finalproject: $(OBJECTS)
	cc $(OBJECTS) -o finalproject
	
scoring.o: scoring.c
	cc -c scoring.c

main.o: main.c scoring.h
	cc -c main.c
  
clean:
	rm -f scoring *.o core *~