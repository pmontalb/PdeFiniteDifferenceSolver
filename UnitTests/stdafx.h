// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#ifdef __linux__

#include <gtest/gtest.h>
#include <gtest/gtest.h>

#elif _WIN32

// Headers for CppUnitTest
#include "CppUnitTest.h"

#endif

// TODO: reference additional headers your program requires here
