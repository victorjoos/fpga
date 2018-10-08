#include "timer.h"
#include <stdlib.h>
#include <stdio.h>

void start_timer(timer_t * timer){
  gettimeofday(&timer->start,NULL);
  return;
}

double stop_timer(timer_t * timer){
  gettimeofday(&timer->stop,NULL);
  return
    (double) (timer->stop.tv_sec-timer->start.tv_sec) +
    (double) (timer->stop.tv_usec-timer->start.tv_usec) / 1e6;
}
void print_stop_timer(timer_t * timer){
  printf("%f \n", stop_timer(timer));
  return;
}
