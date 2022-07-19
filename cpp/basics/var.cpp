// Starndard basic code

#include <iostream>

int a, b = 5; // Global var

/* yes, a large observation */

int main(){
	bool flag; // Local var
	a = 7;
	flag = false;
	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "flag = " << flag << std::endl;
	std::cout << "b - a = " << b-a << std::endl;
	std::cout << "a + b = " << a+b << std::endl;
	unsigned int positive;
	positive = b - a;
	std::cout << "b - a (unsigned) =" << positive << std::endl;
	return(0);
}
