// Using directives

#include <iostream>

#define MAX_VAL 5000
#define DEBUG

int main(){
	int oof = MAX_VAL;
#ifdef DEBUG
	std::cout << "Someone check this assigment!" << std::endl;
#endif
	oof -=1;
	std::cout << "5000 - 1 = " << oof << std::endl;
	return 0;
}
