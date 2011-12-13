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
} gpgpu_timer_t;

gpgpu_timer_t _timer_p;
double uSec;

void gpgpu_timer_init(gpgpu_timer_t *p) {
    p->total.tv_sec = p->total.tv_usec = 0;
}
void gpgpu_timer_start(gpgpu_timer_t *p) {
 gettimeofday(&p->start, NULL);
}
void gpgpu_timer_stop(gpgpu_timer_t *p)
{
    struct timeval stop;
    gettimeofday(&stop, NULL);
    timersub(&stop, &p->start, &stop); /* stop = stop - start */
    timeradd(&stop, &p->total, &p->total); /* total += stop - start */
}
void gpgpu_timer_usec(gpgpu_timer_t *p)
{
    uSec = (double)((p->total.tv_sec) * 1e6 + p->total.tv_usec);
}

// tic, toc support
#define TIC  do {                               \
        gpgpu_timer_init(&_timer_p);            \
        gpgpu_timer_start(&_timer_p);           \
    } while(0)

#define TOC  do {                               \
        gpgpu_timer_stop(&_timer_p);            \
        gpgpu_timer_usec(&_timer_p);            \
    } while (0)
