#include <sys/time.h>

typedef struct timer {
  struct timeval start;
  struct timeval stop;
} timer_t;

void start_timer(timer_t * timer);
double stop_timer(timer_t * timer);
void print_stop_timer(timer_t * timer);
