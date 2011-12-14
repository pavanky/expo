#include <unistd.h>
#ifndef __USE_BSD
#define __USE_BSD
#include <sys/time.h>
#undef __USE_BSD
#else
#include <sys/time.h>
#endif  /* __USE_BSD */
typedef struct {
    int is_initialized;
    struct timeval start, total;
} timer_t;

timer_t _timer_p;
double uSec;

void timer_init(timer_t *p) {
    p->total.tv_sec = p->total.tv_usec = 0;
}
void timer_start(timer_t *p) {
    gettimeofday(&p->start, NULL);
}
void timer_stop(timer_t *p)
{
    struct timeval stop;
    gettimeofday(&stop, NULL);
    timersub(&stop, &p->start, &stop); /* stop = stop - start */
    timeradd(&stop, &p->total, &p->total); /* total += stop - start */
}
void timer_usec(timer_t *p)
{
    uSec = (double)((p->total.tv_sec) * 1e6 + p->total.tv_usec);
}

// tic, toc support
#define TIC  do {                               \
        timer_init(&_timer_p);                  \
        timer_start(&_timer_p);                 \
    } while(0)

#define TOC  do {                               \
        timer_stop(&_timer_p);                  \
        timer_usec(&_timer_p);                  \
    } while (0)
