static char help[] = "Solve a standard 1D FD system with KSP + PC. \n\n";

/*
 * Details
 *
 *
 */

#include <petscksp.h>

// Declare functions
extern PetscErrorCode makeMatrix(Mat*, PetscInt);
// extern PetscErrorCode makeVector(Vec*);
// extern PetscErrorCode makeKSP(KSP*, PC*, Mat*);

// Main section
int main(int argc, char **args){
	Vec 		x, b, u;
	Mat 		A;
	KSP 		ksp;
	// PC			pc;
	PetscReal 	norm;
	PetscInt 	dim=10;//i, col[3], its;
	PetscMPIInt	rank, size; 
	// PetscScalar values[3];
	PetscBool	non_zero_guess = 0;

	// Call PETSc
	PetscCall(PetscInitialize(&argc, &args, (char*)0, help));
	// Recover nro. processes and rank
	PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

	// Set used-defined options
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-non_zero_guess", &non_zero_guess, NULL));
	
	// Make Vectors
	// PetscCall(makeVector(&x));
	PetscCall(VecCreate(PETSC_COMM_SELF,&x));
	PetscCall(PetscObjectSetName((PetscObject) x, "Solution"));
	PetscCall(VecSetSizes(x,PETSC_DECIDE,dim));
	PetscCall(VecSetFromOptions(x));

	PetscCall(VecDuplicate(x,&b));
	PetscCall(VecDuplicate(x,&u));
	if (non_zero_guess) PetscCall(VecSet(x, 1.0));
	
	// Make Matrix
	PetscCall(MatCreate(PETSC_COMM_SELF,&A));
	PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
	PetscCall(MatSetFromOptions(A));
	PetscCall(MatSetUp(A));	

	PetscCall(makeMatrix(&A, dim));

	// TODO Make exact solution
	PetscCall(VecSet(u, -0.5));
	PetscCall(MatMult(A,u,b));

	// Make KSP solver
	// PetscCall(makeKSP(&ksp, &pc, &A));
	PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
	PetscCall(KSPSetOperators(ksp,A,A));
	PetscCall(KSPSetFromOptions(ksp));

	// Solve
	PetscCall(KSPSolve(ksp, b, x));
	
	// Compute norm
	PetscCall(VecAXPY(x,-1.0,u));
	PetscCall(VecNorm(x,NORM_2,&norm));
	PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g \n\n",(double)norm));

	// Destroy
	PetscCall(VecDestroy(&x));
	PetscCall(VecDestroy(&u));
	PetscCall(VecDestroy(&b));
	PetscCall(MatDestroy(&A));
	PetscCall(KSPDestroy(&ksp));

	// Finalize PETSc
	PetscCall(PetscFinalize());
	return 0;
}

// make Functions
PetscErrorCode makeMatrix(Mat* A, PetscInt n){
	PetscInt 	i, Istart, Iend, col[3];
	PetscScalar	values[3], h;

	// Get Local Ownership + set up const
	MatGetOwnershipRange(*A, &Istart, &Iend);
	h = 1/(n-1);
	values[0] = -1; values[1] = 2; values[2] = -1;

	//Insert values
	for (i=Istart; i<Iend; i++){
		col[0] = i-1; col[1] = i; col[2] = i+1;
		if (i>1) {
			PetscCall(MatSetValues(*A,1,&i,1,&(col[0]),&(values[0]),INSERT_VALUES));
		}
		if (i<n-1) {
			PetscCall(MatSetValues(*A,1,&i,1,&(col[2]),&(values[2]),INSERT_VALUES));
		}
		PetscCall(MatSetValues(*A,1,&i,1,&(col[1]),&(values[1]),INSERT_VALUES));
	}

	//Assemble
	PetscCall(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));
	return 0;
}

