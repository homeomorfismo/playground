EXECS=mpi_hello_world
MPICC?=mpicc

all: ${EXECS}

mpi_hello_world: mpi_hello_world.c
	${MPICC} -o mpi_hello_world mpi_hello_world.c

send_recive: send_recive.c
	${MPICC} -o send_recive send_recive.c

clean:
	rm ${EXECS}
