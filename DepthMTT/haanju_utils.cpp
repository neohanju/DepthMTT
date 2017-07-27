#include "haanju_utils.hpp"


/************************************************************************
 Method Name: FormattedString
 Description:
	- Printf function for std::string type.
 Input Arguments:
	- _formatted_string: The formatted string input.
	- ...              : The assigning values of '_formatted_string'.
 Return Values:
	- std::string: The result of the formatted string.
************************************************************************/
std::string hj::FormattedString(const std::string _formatted_string, ...)
{
	int final_n, n = ((int)_formatted_string.size()) * 2; /* Reserve two times as much as the length of the _formatted_string */
	std::string str;
	std::unique_ptr<char[]> formatted;
	va_list ap;
	while (1)
	{
		formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
		strcpy_s(formatted.get(), n, _formatted_string.c_str());
		va_start(ap, _formatted_string);
		final_n = vsnprintf(&formatted[0], n, _formatted_string.c_str(), ap);
		va_end(ap);
		if (final_n < 0 || final_n >= n)
		{
			n += abs(final_n - n + 1);
		}
		else
		{
			break;
		}
	}
	return std::string(formatted.get());
}

//()()
//('')HAANJU.YOO
