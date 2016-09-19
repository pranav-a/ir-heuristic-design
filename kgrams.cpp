#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> kgrams(std::string, int);
std::vector<std::string> cyclic(std::vector<std::string>);

main()
{
	std::string input = "abcde";
	int len = 2;
	std::vector<std::string> result, output;
	result = kgrams(input, len);
	output = cyclic(result);
	for(auto& i : output)
		std::cout << i << std::endl;
}

std::vector<std::string> kgrams(std::string input, int len)
{
	std::vector<std::string> result;
	//This function splits the word into k-grams. Like abc to (a, ab, bc, c)
	
	if (input.length() < len)
		len = input.length();
	
	for(int i = 1; i != len; i++)
		result.push_back(input.substr(0, i)); // Add starting combinations
	
	for(int i = 0; i != input.length() - len; i++)
		result.push_back(input.substr(i, len)); // Add k-grams
	
	for(int i = len; i != 0; i --)
		result.push_back(input.substr(input.length() - i, i)); // Add ending combinations
	
	return result;
}

std::vector<std::string> cyclic(std::vector<std::string> input)
{
	int len = input.size();
	std::vector<std::string> result;
	
	// No offset
	for(int i = 1; i <= len; i++)
	{
		if(i <= (len - i + 1))
			result.push_back(std::to_string(i) + input[i - 1]); // Append ahead the numbers from start
		else
			result.push_back(input[i - 1] + std::to_string(len - i + 1)); // Append behind the numbers from end
	}
	
	// Offset of -1
	for(int i = 0; i <= len - 1; i++)
	{
		if(i <= (len - i - 1))
			result.push_back(std::to_string(i) + input[i]);
		else
			result.push_back(input[i] + std::to_string(len - i - 1));
	}
	
	//Offset of 1
	for(int i = 2; i <= len + 1; i++)
	{
		if(i <= (len - i + 3))
			result.push_back(std::to_string(i) + input[i - 2]);
		else
			result.push_back(input[i - 2] + std::to_string(len - i + 3));
	}
	return result;
}
