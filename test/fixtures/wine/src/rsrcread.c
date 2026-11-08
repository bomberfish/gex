/* What a shell does with a program put on the desktop: its icons and its
 * version (argv[1]: the file), each timed, then the file as an image
 * resource. The icons are read twice, the second time timed as "icons":
 * the first read also pays the first icon's making (seconds on gex, and
 * varying) and the file's first read (a page reads a file's bytes in as
 * they are first asked for). Says "done" at the end. */
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    const char *f = argc > 1 ? argv[1] : "big.exe";
    HICON big = 0, small = 0;
    DWORD t, h, n;
    UINT got, len;
    char *buf, *desc = NULL;
    HMODULE m;

    setvbuf(stdout, NULL, _IONBF, 0);
    t = GetTickCount();
    got = ExtractIconExA(f, 0, &big, &small, 1);
    printf("icons first %u: %lu ms\n", got, GetTickCount() - t);
    if (big) DestroyIcon(big);
    if (small) DestroyIcon(small);
    big = small = 0;
    t = GetTickCount();
    got = ExtractIconExA(f, 0, &big, &small, 1);
    printf("icons %u %s: %lu ms\n", got, big && small ? "both" : big || small ? "one" : "none", GetTickCount() - t);
    if (big) DestroyIcon(big);
    if (small) DestroyIcon(small);
    t = GetTickCount();
    n = GetFileVersionInfoSizeA(f, &h);
    printf("version size %lu: %lu ms\n", n, GetTickCount() - t);
    if (n && (buf = malloc(n)))
    {
        t = GetTickCount();
        if (GetFileVersionInfoA(f, 0, n, buf))
            VerQueryValueA(buf, "\\StringFileInfo\\040904b0\\FileDescription", (void **)&desc, &len);
        printf("version \"%s\": %lu ms\n", desc ? desc : "", GetTickCount() - t);
        free(buf);
    }
    t = GetTickCount();
    m = LoadLibraryExA(f, 0, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    printf("image resource %s: %lu ms\n", m ? "loaded" : "refused", GetTickCount() - t);
    if (m) FreeLibrary(m);
    printf("done\n");
    return 0;
}
