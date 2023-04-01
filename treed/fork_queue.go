package treed

import (
	"runtime"
	"sync/atomic"
)

type forkQueueTask[T any] struct {
	Started int32
	F       func() T
	Result  chan T
}

// A forkQueue lets you run recursive tasks with a fixed maximum number of
// Goroutines. To use the queue, create with newForkQueue(), then run the root
// task with Run(). Within a task, you may call Fork() to run two sub-tasks,
// potentially allowing separate goroutines to run the two sub-tasks.
type forkQueue[T any] struct {
	queue chan *forkQueueTask[T]
}

func newForkQueue[T any](numWorkers int) *forkQueue[T] {
	if numWorkers == 0 {
		numWorkers = runtime.GOMAXPROCS(0)
	}
	res := &forkQueue[T]{
		queue: make(chan *forkQueueTask[T], numWorkers*1000),
	}
	for i := 0; i < numWorkers; i++ {
		go res.worker()
	}
	return res
}

func (f *forkQueue[T]) Run(fn func() T) T {
	defer close(f.queue)
	task := &forkQueueTask[T]{F: fn, Result: make(chan T, 1)}
	f.queue <- task
	return <-task.Result
}

func (f *forkQueue[T]) Fork(fn1, fn2 func() T) (T, T) {
	task := &forkQueueTask[T]{F: fn2, Result: make(chan T, 1)}
	select {
	case f.queue <- task:
	default:
		// Prevent unbounded memory growth by running the task locally.
		task.Started = 1
		task.Result <- task.F()
	}
	result1 := fn1()
	var result2 T
	if atomic.SwapInt32(&task.Started, 1) == 0 {
		// Task hasn't been started yet, so we kick it off
		// on this worker.
		result2 = fn2()
	} else {
		result2 = <-task.Result
	}
	return result1, result2
}

func (f *forkQueue[T]) worker() {
	for task := range f.queue {
		if atomic.SwapInt32(&task.Started, 1) != 0 {
			// Task already started on a different thread.
			continue
		}
		task.Result <- task.F()
	}
}
