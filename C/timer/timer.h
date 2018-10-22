#include <sys/time.h>

typedef struct timer {
  struct timeval start;
  struct timeval stop;
} my_timer_t;

void start_timer(my_timer_t * timer);
double stop_timer(my_timer_t * timer);
void print_stop_timer(my_timer_t * timer);
