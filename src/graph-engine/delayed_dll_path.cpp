/*
 * Copyright(C) 2023 Advanced Micro Devices, Inc.All Rights Reserved.
*/

#include <windows.h>

namespace {

    struct dllpath
    {
        dllpath()
        {
            ::SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            ::AddDllDirectory(L"C:\\Windows\\System32\\AMD");
        }
    };

    static dllpath x;

}
