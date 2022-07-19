// Mirror exercise: receive an string and print it

#include <iostream>

int main(){
	std::string str;
	std::cout << "Please, write your name below..." << std::endl;
	std::cin >> str;
	std::cout << "This is your name:" << std::endl;
	std::cout << str;
	return 0;
}
