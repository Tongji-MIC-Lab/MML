#ifndef _TXT_PRASE_
#define _TXT_PRASE_
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

class Object
{
public:
    //label
    int lOrg;
    //video id
	string id;
};
class CObjList
{
public:
	vector<Object> _vec;
	bool Init(const char* fileName) 
	{
		_vec.clear();
		bool bReturn = false;
		ifstream gtfile(fileName);
		if (!gtfile.is_open())
		{
			printf("[ERROR]could not open text file %s\n",fileName);
			exit(-1);
		}

		while (!gtfile.eof())
		{
            string line;
            getline(gtfile,line);
            if (0 == line.length())
			{
				break;
			}

            istringstream iss(line);
			if (!iss.fail())
			{
				Object obj;
                iss >> obj.id >> obj.lOrg;
                if (obj.id.length()>0)
                {
                    _vec.push_back(obj);
                    bReturn = true;
                }
                else
                    break;

			} else 
			{
				if (!gtfile.eof()) 
				{
					printf("[ERROR] Error parsing text file.\n");
					exit(-1);
				}
            }
		}
		gtfile.close();
		return bReturn;
	}
};

#endif
