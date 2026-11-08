/* 150 threads in one process, all alive at once: each starts, says so and
 * waits, and once every one has started they are let go and joined. In a
 * browser tab each thread is a Worker, and a memory per Worker ran a tab out
 * of them (about 124) before half had started. Prints how many started. */
#include <windows.h>
#include <stdio.h>

#define N 150

static HANDLE go;
static volatile LONG started;

static DWORD WINAPI one(void *arg)
{
    (void)arg;
    InterlockedIncrement(&started);
    WaitForSingleObject(go, INFINITE);
    return 0;
}

int main(void)
{
    HANDLE t[N];
    int i, n = 0;

    go = CreateEventW(NULL, TRUE, FALSE, NULL);
    for (i = 0; i < N; i++)
    {
        t[i] = CreateThread(NULL, 64 * 1024, one, NULL, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!t[i]) break;
        n++;
    }
    while (started < n) Sleep(10);
    printf("threads150 started %ld of %d\n", started, N);
    fflush(stdout);
    SetEvent(go);
    for (i = 0; i < n; i++)
    {
        WaitForSingleObject(t[i], INFINITE);
        CloseHandle(t[i]);
    }
    printf("threads150 done\n");
    return n == N ? 0 : 1;
}
