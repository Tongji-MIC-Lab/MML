#ifndef COMMON_H
#define COMMON_H
#include <cstdarg>
#include <cstdio>
#include <string>
#include <sys/stat.h>
namespace yi {

static string format(const char* fmt, ...)
{
    va_list vl;

    va_start(vl, fmt);
    int size = vsnprintf(0, 0, fmt, vl) + sizeof('\0');
    va_end(vl);

    char buffer[size];

    va_start(vl, fmt);
    size = vsnprintf(buffer, size, fmt, vl);
    va_end(vl);

    return string(buffer, size);
}


static void makeDir(const string& dir)
{
    mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

}

#endif // COMMON_H
