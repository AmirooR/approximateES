#pragma once

#include <boost/shared_array.hpp>

struct Short
{
    short x;

    Short()
    {
        x = 0;
    }

    Short(short _x):x(_x)
    {
    }

    ~Short()
    {
    }

    Short& operator=(short _x)
    {
        x = _x;
        return *this;
    }

    bool operator==(Short& s){return s.x == x;}
    bool operator!=(Short& s){return s.x != x;}

};

typedef boost::shared_array<short> short_array;

/*bool operator!=(const short_array& s1, const short_array& s2)
{
    for(size_t i = 0; i < 10; i++)
    {
        if( s1[i] != s2[i] )
            return true;
    }
    return false;

}

bool operator==(const short_array& s1, const short_array& s2)
{
    for(size_t i = 0; i < 10; i++)
    {
        if( s1[i] != s2[i] )
            return false;
    }
    return true;

}*/

