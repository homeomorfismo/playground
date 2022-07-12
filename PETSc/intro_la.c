static char help[] = "Solves a tridiag system with KSP.\n\n";
#include<petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv){
	PetscErrorCode ierr;
	PetscMPIInt rank, size;
	PetscInt n = 10, i;	// Matrix size, index (tri)diag
	PetscInt col[3], its;
	PetscScalar value[3];
       	PetscReal norm;
	PetscScalar neg_one = -1.0, one = 1.0;
	PetscBool nonzeroguess = PETSC_FALSE;
	// PetscReal norm;		// Norm of the error

	PetscInitialize(&argc,&argv,NULL,"Setting up some matrices and stuff, with PETSc.\n\n");
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"Uniprocessor example!");
	ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetBool(NULL,NULL,"-nonzero_guess",&nonzeroguess,NULL);CHKERRQ(ierr);

	// Create matrix
	Mat A;
	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	// Set up matrix A
	ierr = MatSetUp(A); CHKERRQ(ierr);
	value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
	for(i=1; i<n-1; i++){
		col[0] = i-1; col[1] = i; col[2] = i+1;
		ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
	}
	i = n-1; col[0] = n-2; col[1] = n-1;
	ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
	i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
	ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	// Create vectors
	Vec x, b, u;

	ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);CHKERRQ(ierr);
	// Duplicate x onto b,u
	ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

	// Set RHS
	ierr = VecSet(u,one);CHKERRQ(ierr);
	ierr = MatMult(A,u,b);CHKERRQ(ierr);

	// Set KSP
	KSP ksp;
	PC pc;

	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
	ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
	ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

	// Runtime options
	ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

	if (nonzeroguess){
		PetscScalar p= 0.5;
		ierr = VecSet(x,p);CHKERRQ(ierr);
		ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);CHKERRQ(ierr);
	}

	// solving the system
	
	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	ierr = VecAXPY(x,neg_one,u);CHKERRQ(ierr);
	ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, iterations %D\n", (double)norm,its);CHKERRQ(ierr);

	// Free space
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&u);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

	ierr = PetscFinalize();

	return 0;
}
