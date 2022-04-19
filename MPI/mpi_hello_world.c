#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){
	// Initialize the MPI env
	MPI_Init(NULL,NULL);
	
	// Get number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process (i.e., like the ID)
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	printf("Hello world form processor %s, rank %d out of %d processors\n",
			processor_name, world_rank, world_size);

	MPI_Finalize();
}

